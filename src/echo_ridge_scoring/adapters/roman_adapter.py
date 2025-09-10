"""
Roman's agentic search backend adapter for Echo Ridge scoring.

Converts Roman's PlaceNorm + WebSnapshot records to Echo Ridge CompanySchema
using only observed values with conservative extractors.
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..schema import (
    CompanySchema, DigitalSchema, OpsSchema, InfoFlowSchema, 
    MarketSchema, BudgetSchema, MetaSchema
)


def to_company_schema(roman_record: Dict[str, Any]) -> Tuple[CompanySchema, List[str]]:
    """
    Convert Roman's PlaceNorm record to Echo Ridge CompanySchema.
    
    Args:
        roman_record: Dict containing 'place_norm' and optionally 'web_snapshot'
        
    Returns:
        Tuple of (CompanySchema, warnings_list)
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    warnings = []
    
    # Extract place and snapshot data
    place_data = roman_record.get('place_norm', roman_record)
    web_snapshot = roman_record.get('web_snapshot')
    
    # Required field mapping (observed values only)
    try:
        company_id = str(place_data['entity_id']) or place_data.get("source_id")
    except KeyError:
        raise ValueError("Missing required field: entity_id or source_id")
    
    # Domain extraction with fallback
    domain = _extract_domain(place_data, warnings)
    
    # Meta fields (observed values)
    meta = _extract_meta_fields(place_data, warnings)
    
    # Digital fields (conservative extraction)
    digital = _extract_digital_fields(place_data, web_snapshot, warnings)
    
    # Ops fields (conservative extraction)  
    ops = _extract_ops_fields(place_data, web_snapshot, warnings)
    
    # Info flow fields (conservative extraction)
    info_flow = _extract_info_flow_fields(place_data, web_snapshot, warnings)
    
    # Market fields (conservative extraction)
    market = _extract_market_fields(place_data, web_snapshot, warnings)
    
    # Budget fields (conservative extraction)
    budget = _extract_budget_fields(place_data, web_snapshot, warnings)
    
    # Build CompanySchema
    try:
        company = CompanySchema(
            company_id=company_id,
            domain=domain,
            digital=digital,
            ops=ops,
            info_flow=info_flow,
            market=market,
            budget=budget,
            meta=meta
        )
        return company, warnings
    except Exception as e:
        raise ValueError(f"Failed to create CompanySchema: {e}")


def _extract_domain(place_data: Dict[str, Any], warnings: List[str]) -> str:
    """Extract domain from website URL or metadata."""
    # Try metadata.domain first (explicit)
    if 'metadata' in place_data and 'domain' in place_data['metadata']:
        domain = place_data['metadata']['domain']
        if domain and isinstance(domain, str):
            return domain.lower().strip()
    
    # Fallback to website URL parsing
    website = place_data.get('website')
    if website and isinstance(website, str):
        try:
            parsed = urlparse(website.strip())
            if parsed.netloc:
                domain = parsed.netloc.lower()
                # Remove www. prefix
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
        except Exception:
            pass
    
    # Last resort: use entity_id as domain proxy
    entity_id = place_data.get('entity_id', 'unknown')
    warnings.append(f"No valid domain found, using entity_id as proxy: {entity_id}")
    return f"{entity_id}.placeholder.com"


def _extract_meta_fields(place_data: Dict[str, Any], warnings: List[str]) -> MetaSchema:
    """Extract metadata fields from place data."""
    # Scrape timestamp
    scrape_ts_str = place_data.get('created_at')
    if scrape_ts_str:
        try:
            # Parse ISO timestamp
            if isinstance(scrape_ts_str, str):
                # Remove timezone info for parsing if present
                cleaned_ts = re.sub(r'\+00:00$', 'Z', scrape_ts_str)
                cleaned_ts = cleaned_ts.replace('Z', '+00:00')
                scrape_ts = datetime.fromisoformat(cleaned_ts.replace('Z', '+00:00'))
            else:
                scrape_ts = datetime.now()
                warnings.append("Invalid created_at format, using current time")
        except Exception:
            scrape_ts = datetime.now()
            warnings.append("Failed to parse created_at, using current time")
    else:
        scrape_ts = datetime.now()
        warnings.append("Missing created_at field, using current time")
    
    # Source confidence
    source_confidence = place_data.get('confidence_score', 0.5)
    if not isinstance(source_confidence, (int, float)) or not (0 <= source_confidence <= 1):
        source_confidence = 0.5
        warnings.append("Invalid or missing confidence_score, defaulting to 0.5")
    
    return MetaSchema(
        scrape_ts=scrape_ts,
        source_confidence=float(source_confidence)
    )


def _extract_digital_fields(place_data: Dict[str, Any], web_snapshot: Optional[Dict[str, Any]], 
                          warnings: List[str]) -> DigitalSchema:
    """Extract digital maturity fields using conservative detection."""
    # Default null values
    pagespeed = 50  # Neutral default
    crm_flag = False
    ecom_flag = False
    
    warnings.append("digital.pagespeed set to neutral default (50) - no pagespeed data available")
    
    if web_snapshot and 'pages' in web_snapshot:
        # Conservative CRM/ecom detection from website content
        all_text = ""
        for page in web_snapshot['pages']:
            page_text = page.get('text', '')
            if isinstance(page_text, str):
                all_text += page_text.lower()
        
        # CRM detection (high-precision markers)
        crm_markers = [
            'customer portal', 'client portal', 'member portal', 
            'login to your account', 'member login', 'client login',
            'dashboard', 'crm', 'salesforce', 'hubspot'
        ]
        if any(marker in all_text for marker in crm_markers):
            crm_flag = True
        else:
            warnings.append("digital.crm_flag set to False - no unambiguous CRM markers detected")
        
        # E-commerce detection (high-precision markers)
        ecom_markers = [
            'add to cart', 'shopping cart', 'checkout', 'buy now',
            'order now', 'purchase', 'payment', 'stripe', 'paypal',
            'online store', 'e-commerce', 'ecommerce'
        ]
        if any(marker in all_text for marker in ecom_markers):
            ecom_flag = True
        else:
            warnings.append("digital.ecom_flag set to False - no unambiguous e-commerce markers detected")
    else:
        warnings.append("digital.crm_flag and ecom_flag set to False - no website content available")
    
    return DigitalSchema(
        pagespeed=pagespeed,
        crm_flag=crm_flag,
        ecom_flag=ecom_flag
    )


def _extract_ops_fields(place_data: Dict[str, Any], web_snapshot: Optional[Dict[str, Any]], 
                       warnings: List[str]) -> OpsSchema:
    """Extract operations fields using conservative extraction."""
    employees = 1  # Minimum viable
    locations = 1  # Default single location
    services_count = 1  # Minimum viable
    
    # Try to count locations from address data
    if 'address' in place_data:
        locations = 1  # Single location confirmed
    
    # Try to extract from provenance (multiple sources = multiple locations)
    provenance = place_data.get('provenance', [])
    if isinstance(provenance, list) and len(provenance) > 1:
        unique_locations = set()
        for prov in provenance:
            if isinstance(prov, dict) and 'source_id' in prov:
                unique_locations.add(prov['source_id'])
        if len(unique_locations) > 1:
            locations = len(unique_locations)
        else:
            warnings.append("ops.locations set to 1 - single location detected from provenance")
    else:
        warnings.append("ops.locations set to 1 - single location assumed")
    
    # Services extraction from website content
    if web_snapshot and 'pages' in web_snapshot:
        all_text = ""
        for page in web_snapshot['pages']:
            page_text = page.get('text', '')
            if isinstance(page_text, str):
                all_text += page_text.lower()
        
        # Look for explicit service lists or counts
        service_patterns = [
            r'services?:\s*(\d+)',
            r'(\d+)\s+services?',
            r'we offer\s+(\d+)',
        ]
        
        for pattern in service_patterns:
            match = re.search(pattern, all_text)
            if match:
                try:
                    services_count = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        else:
            warnings.append("ops.services_count set to 1 - no explicit service count found")
    else:
        warnings.append("ops.services_count set to 1 - no website content available")
    
    warnings.append("ops.employees set to 1 - no employee count data available")
    
    return OpsSchema(
        employees=employees,
        locations=locations,
        services_count=services_count
    )


def _extract_info_flow_fields(place_data: Dict[str, Any], web_snapshot: Optional[Dict[str, Any]], 
                            warnings: List[str]) -> InfoFlowSchema:
    """Extract information flow fields."""
    daily_docs_est = 10  # Minimal default
    warnings.append("info_flow.daily_docs_est set to minimal default (10) - no document volume data available")
    
    return InfoFlowSchema(daily_docs_est=daily_docs_est)


def _extract_market_fields(place_data: Dict[str, Any], web_snapshot: Optional[Dict[str, Any]], 
                         warnings: List[str]) -> MarketSchema:
    """Extract market analysis fields."""
    competitor_density = 5  # Neutral default
    industry_growth_pct = 2.0  # Neutral default
    rivalry_index = 0.5  # Neutral default
    
    warnings.extend([
        "market.competitor_density set to neutral default (5) - no competitor data available",
        "market.industry_growth_pct set to neutral default (2.0) - no growth data available", 
        "market.rivalry_index set to neutral default (0.5) - no rivalry data available"
    ])
    
    return MarketSchema(
        competitor_density=competitor_density,
        industry_growth_pct=industry_growth_pct,
        rivalry_index=rivalry_index
    )


def _extract_budget_fields(place_data: Dict[str, Any], web_snapshot: Optional[Dict[str, Any]], 
                         warnings: List[str]) -> BudgetSchema:
    """Extract budget/revenue fields."""
    revenue_est_usd = 100000.0  # Minimal viable business default
    
    # Try to extract from website pricing if available
    if web_snapshot and 'pages' in web_snapshot:
        all_text = ""
        for page in web_snapshot['pages']:
            page_text = page.get('text', '')
            if isinstance(page_text, str):
                all_text += page_text.lower()
        
        # Look for explicit pricing patterns
        price_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*)\s*(?:per|/)',
            r'revenue:?\s*\$?(\d{1,3}(?:,\d{3})*)',
            r'annual:?\s*\$?(\d{1,3}(?:,\d{3})*)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                try:
                    # Use highest price found as revenue proxy
                    prices = [int(match.replace(',', '')) for match in matches]
                    if prices:
                        max_price = max(prices)
                        # Estimate annual revenue (conservative multiplier)
                        revenue_est_usd = max_price * 12 * 10  # Monthly price * 12 * 10 customers
                        break
                except (ValueError, TypeError):
                    continue
        else:
            warnings.append("budget.revenue_est_usd set to minimal default (100000) - no pricing data found")
    else:
        warnings.append("budget.revenue_est_usd set to minimal default (100000) - no website content available")
    
    return BudgetSchema(revenue_est_usd=float(revenue_est_usd))