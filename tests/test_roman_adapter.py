"""
Tests for Roman adapter using real JSONL data samples.
"""

import json
import pytest
from datetime import datetime
from src.echo_ridge_scoring.adapters.roman_adapter import to_company_schema
from src.echo_ridge_scoring.schema import CompanySchema


# Real Roman JSONL samples from examples/echoridge_search_backend/
SAMPLE_PLACE_NORM = {
    "entity_id": "a111c9c9-d037-4a8d-aeb7-65d8dd925a44",
    "name": "Princeton Club Xpress Monona",
    "category": "business",
    "address": {
        "line1": "5413 Monona Dr",
        "line2": None,
        "city": "WI 53716",
        "region": "USA", 
        "postal": None,
        "country": "US",
        "formatted": "5413 Monona Dr, Monona, WI 53716, USA"
    },
    "lat": 43.0605392,
    "lon": -89.32649230000001,
    "phone": "+16086632639",
    "website": "https://www.princetonclub.net/xpress-monona",
    "emails": [],
    "metadata": {
        "rating": 4.3,
        "ratings_count": 52,
        "business_status": "OPERATIONAL",
        "price_level": None,
        "types": ["establishment", "gym", "health", "point_of_interest"],
        "domain": "princetonclub.net"
    },
    "provenance": [
        {
            "source": "google_places",
            "source_id": "ChIJbdPgENRTBogRh7xBDemepDY", 
            "fetched_at": "2025-09-03T23:13:19.678832+00:00",
            "confidence": 0.9999999999999999
        }
    ],
    "confidence_score": 0.9999999999999999,
    "is_duplicate": False,
    "master_id": None,
    "created_at": "2025-09-03T23:13:19.679721+00:00",
    "updated_at": "2025-09-03T23:13:19.679722+00:00"
}

SAMPLE_WEB_SNAPSHOT = {
    "entity_id": "3717b7ee-a712-4595-9d73-bbde6fc49de4",
    "website": "https://www.anytimefitness.com/locations/madison-wisconsin-2104",
    "crawl_id": "ec19cdd4-bd41-471b-8b8f-4aa557246a8d",
    "pages": [{
        "url": "https://www.anytimefitness.com/locations/madison-wisconsin-2104",
        "title": "Anytime Fitness - Gym in Madison, Wisconsin, 53715",
        "text": "Join for $1 LEARN MORE Open to Members 24/7 Anytime Fitness gym in Madison E Campus Mall 301 East Campus Mall Suite 203 Madison Wisconsin 53715 (608) 237-2717 Contact Us TRY US FREE JOIN NOW Explore Memberships customer portal member login dashboard checkout purchase payment stripe 24-Month Plan $19.99 Due Bi-weekly SELECT 12-Month Plan $22.99 Due Bi-weekly SELECT"
    }],
    "total_pages": 1,
    "crawl_status": "success",
    "fetched_at": "2025-09-03T23:13:42+00:00"
}

SAMPLE_AI_SCORE = {
    "entity_id": "a111c9c9-d037-4a8d-aeb7-65d8dd925a44",
    "name": "Princeton Club Xpress Monona", 
    "category": "business",
    "website": "https://www.princetonclub.net/xpress-monona",
    "scored_at": "2025-09-03T23:13:42.254107+00:00",
    "model": "gpt-4o",
    "overall_note": "Princeton Club Xpress Monona demonstrates moderate digital maturity.",
    "dimb_scores": {
        "D": {
            "component": "D",
            "full_name": "Digital Maturity", 
            "value": 0.6,
            "evidence": "Website with membership options",
            "confidence": 0.8
        },
        "O": {
            "component": "O",
            "full_name": "Operational Complexity",
            "value": 0.4, 
            "evidence": "Multiple locations, fitness services",
            "confidence": 0.8
        },
        "I": {
            "component": "I",
            "full_name": "Information Flow",
            "value": 0.3,
            "evidence": "Limited evidence of complex systems",
            "confidence": 0.8
        },
        "M": {
            "component": "M", 
            "full_name": "Market Pressure",
            "value": 0.5,
            "evidence": "Competitive fitness market",
            "confidence": 0.8
        },
        "B": {
            "component": "B",
            "full_name": "Budget Signals", 
            "value": 0.5,
            "evidence": "Mid-tier pricing with premium features",
            "confidence": 0.8
        }
    },
    "overall_score": 0.465,
    "score_summary": {
        "digital_maturity": 0.6,
        "operational_complexity": 0.4,
        "information_flow": 0.3,
        "market_pressure": 0.5,
        "budget_signals": 0.5,
        "weighted_overall": 0.465
    }
}


class TestRomanAdapter:
    """Test Roman adapter with real JSONL samples."""
    
    def test_basic_place_norm_conversion(self):
        """Test basic PlaceNorm to CompanySchema conversion."""
        company, warnings = to_company_schema(SAMPLE_PLACE_NORM)
        
        # Verify basic mapping
        assert company.company_id == "a111c9c9-d037-4a8d-aeb7-65d8dd925a44"
        assert company.domain == "princetonclub.net"
        assert company.meta.source_confidence == 0.9999999999999999
        assert isinstance(company.meta.scrape_ts, datetime)
        
        # Verify warnings for missing fields
        assert len(warnings) > 0
        warning_text = " ".join(warnings)
        assert "pagespeed" in warning_text
        assert "employees" in warning_text
        
    def test_domain_extraction_from_metadata(self):
        """Test domain extraction from metadata.domain field."""
        company, _ = to_company_schema(SAMPLE_PLACE_NORM)
        assert company.domain == "princetonclub.net"
        
    def test_domain_extraction_from_website_url(self):
        """Test domain extraction fallback to website URL."""
        sample = SAMPLE_PLACE_NORM.copy()
        del sample["metadata"]["domain"]  # Remove explicit domain
        
        company, warnings = to_company_schema(sample)
        assert company.domain == "princetonclub.net"  # From website URL
        
    def test_missing_domain_fallback(self):
        """Test fallback when no domain or website available.""" 
        sample = SAMPLE_PLACE_NORM.copy()
        del sample["metadata"]["domain"]
        del sample["website"] 
        
        company, warnings = to_company_schema(sample)
        assert company.domain == "a111c9c9-d037-4a8d-aeb7-65d8dd925a44.placeholder.com"
        assert any("No valid domain found" in w for w in warnings)
        
    def test_web_snapshot_crm_detection(self):
        """Test CRM detection from WebSnapshot content."""
        record = {
            "place_norm": SAMPLE_PLACE_NORM,
            "web_snapshot": SAMPLE_WEB_SNAPSHOT
        }
        
        company, warnings = to_company_schema(record)
        
        # Should detect CRM markers (customer portal, member login, dashboard)
        assert company.digital.crm_flag == True
        
    def test_web_snapshot_ecom_detection(self):
        """Test e-commerce detection from WebSnapshot content."""
        record = {
            "place_norm": SAMPLE_PLACE_NORM, 
            "web_snapshot": SAMPLE_WEB_SNAPSHOT
        }
        
        company, warnings = to_company_schema(record)
        
        # Should detect ecom markers (checkout, purchase, payment, stripe)
        assert company.digital.ecom_flag == True
        
    def test_conservative_extraction_defaults(self):
        """Test that conservative extraction provides reasonable defaults."""
        company, warnings = to_company_schema(SAMPLE_PLACE_NORM)
        
        # Digital defaults
        assert company.digital.pagespeed == 50  # Neutral default
        
        # Ops defaults
        assert company.ops.employees == 1  # Minimal viable
        assert company.ops.locations == 1  # Single location
        assert company.ops.services_count == 1  # Minimal viable
        
        # Info flow defaults
        assert company.info_flow.daily_docs_est == 10  # Minimal default
        
        # Market defaults (neutral)
        assert company.market.competitor_density == 5
        assert company.market.industry_growth_pct == 2.0
        assert company.market.rivalry_index == 0.5
        
        # Budget defaults
        assert company.budget.revenue_est_usd == 100000.0  # Minimal viable
        
    def test_multiple_locations_from_provenance(self):
        """Test location counting from provenance data.""" 
        sample = SAMPLE_PLACE_NORM.copy()
        # Add multiple unique sources
        sample["provenance"] = [
            {"source": "google_places", "source_id": "id1", "confidence": 0.9},
            {"source": "google_places", "source_id": "id2", "confidence": 0.9},
            {"source": "google_places", "source_id": "id3", "confidence": 0.9}
        ]
        
        company, warnings = to_company_schema(sample)
        assert company.ops.locations == 3
        
    def test_invalid_input_handling(self):
        """Test handling of invalid or missing required fields."""
        # Missing entity_id
        invalid_sample = SAMPLE_PLACE_NORM.copy()
        del invalid_sample["entity_id"]
        
        with pytest.raises(ValueError, match="Missing required field: entity_id"):
            to_company_schema(invalid_sample)
            
    def test_warning_generation(self):
        """Test comprehensive warning generation for missing data."""
        company, warnings = to_company_schema(SAMPLE_PLACE_NORM)
        
        # Should have warnings for all unextractable fields
        warning_keywords = [
            "pagespeed", "employees", "daily_docs_est", "competitor_density", 
            "industry_growth_pct", "rivalry_index", "revenue_est_usd"
        ]
        
        warning_text = " ".join(warnings).lower()
        for keyword in warning_keywords:
            assert keyword in warning_text, f"Missing warning for {keyword}"
            
    def test_timestamp_parsing(self):
        """Test various timestamp formats."""
        # Valid ISO timestamp
        company, warnings = to_company_schema(SAMPLE_PLACE_NORM)
        assert isinstance(company.meta.scrape_ts, datetime)
        
        # Invalid timestamp format
        sample = SAMPLE_PLACE_NORM.copy()
        sample["created_at"] = "invalid-timestamp"
        
        company, warnings = to_company_schema(sample)
        assert isinstance(company.meta.scrape_ts, datetime)
        assert any("Failed to parse created_at" in w for w in warnings)
        
    def test_confidence_score_validation(self):
        """Test confidence score validation and defaults."""
        # Valid confidence
        company, _ = to_company_schema(SAMPLE_PLACE_NORM)
        assert company.meta.source_confidence == 0.9999999999999999
        
        # Invalid confidence (out of range)
        sample = SAMPLE_PLACE_NORM.copy()
        sample["confidence_score"] = 1.5  # > 1.0
        
        company, warnings = to_company_schema(sample)
        assert company.meta.source_confidence == 0.5  # Default
        assert any("Invalid or missing confidence_score" in w for w in warnings)
        
    def test_pricing_extraction_from_web_content(self):
        """Test revenue estimation from website pricing data."""
        # Web content with pricing
        pricing_snapshot = SAMPLE_WEB_SNAPSHOT.copy()
        pricing_snapshot["pages"][0]["text"] += " Annual membership: $500 per year Revenue: $1,000,000"
        
        record = {
            "place_norm": SAMPLE_PLACE_NORM,
            "web_snapshot": pricing_snapshot
        }
        
        company, warnings = to_company_schema(record)
        
        # Should extract revenue from pricing patterns
        assert company.budget.revenue_est_usd > 100000.0  # More than default
        
    def test_end_to_end_integration(self):
        """Test complete end-to-end conversion with all data."""
        record = {
            "place_norm": SAMPLE_PLACE_NORM,
            "web_snapshot": SAMPLE_WEB_SNAPSHOT  
        }
        
        company, warnings = to_company_schema(record)
        
        # Verify complete CompanySchema is created
        assert isinstance(company, CompanySchema)
        assert company.company_id
        assert company.domain
        assert company.digital
        assert company.ops
        assert company.info_flow
        assert company.market
        assert company.budget
        assert company.meta
        
        # Verify warnings are comprehensive but not excessive
        assert 5 <= len(warnings) <= 15  # Reasonable warning count