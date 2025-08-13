"""
Example usage of the Echo Ridge Scoring System

This demonstrates how to use the complete DOIMB scoring framework
to evaluate company data and generate comprehensive scores.
"""

from datetime import datetime
from src.schema import *
from src.normalization import NormContext
from src.scoring import SubscoreCalculator, FinalScorer


def create_sample_companies():
    """Create sample companies for demonstration"""
    
    companies = [
        # High-performing tech company
        CompanySchema(
            company_id="tech_leader_001",
            domain="techleader.com",
            digital=DigitalSchema(pagespeed=95, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=250, locations=5, services_count=25),
            info_flow=InfoFlowSchema(daily_docs_est=500),
            market=MarketSchema(competitor_density=30, industry_growth_pct=15.5, rivalry_index=0.4),
            budget=BudgetSchema(revenue_est_usd=15000000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.95)
        ),
        
        # Mid-size service company
        CompanySchema(
            company_id="service_mid_002",
            domain="servicemid.com",
            digital=DigitalSchema(pagespeed=72, crm_flag=True, ecom_flag=False),
            ops=OpsSchema(employees=50, locations=3, services_count=12),
            info_flow=InfoFlowSchema(daily_docs_est=100),
            market=MarketSchema(competitor_density=15, industry_growth_pct=3.2, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=2500000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
        ),
        
        # Small traditional business
        CompanySchema(
            company_id="small_trad_003",
            domain="smallbiz.com",
            digital=DigitalSchema(pagespeed=45, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=8, locations=1, services_count=5),
            info_flow=InfoFlowSchema(daily_docs_est=25),
            market=MarketSchema(competitor_density=5, industry_growth_pct=-1.5, rivalry_index=0.8),
            budget=BudgetSchema(revenue_est_usd=350000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.75)
        ),
        
        # Large enterprise
        CompanySchema(
            company_id="enterprise_004",
            domain="bigcorp.com",
            digital=DigitalSchema(pagespeed=88, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=2500, locations=25, services_count=45),
            info_flow=InfoFlowSchema(daily_docs_est=2000),
            market=MarketSchema(competitor_density=50, industry_growth_pct=8.7, rivalry_index=0.3),
            budget=BudgetSchema(revenue_est_usd=125000000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.92)
        )
    ]
    
    return companies


def main():
    """Demonstrate the complete scoring workflow"""
    
    print("=== Echo Ridge Scoring System Demo ===\n")
    
    # Create sample companies
    companies = create_sample_companies()
    print(f"Created {len(companies)} sample companies for evaluation\n")
    
    # Initialize and fit normalization context
    print("1. Fitting normalization context...")
    norm_context = NormContext(confidence_threshold=0.7)
    norm_context.fit(companies)
    print("   ✓ Normalization parameters calculated from company data\n")
    
    # Initialize scoring system
    print("2. Initializing scoring system...")
    final_scorer = FinalScorer()
    print("   ✓ DOIMB scoring framework ready\n")
    
    # Score each company
    print("3. Calculating scores for all companies...\n")
    
    results = []

    for company in companies:
        subscore_calc = SubscoreCalculator(norm_context)
        subscores = subscore_calc.calculate_subscores(company)
        result = final_scorer.score(subscores)
        results.append((company, result))
        
        print(f"--- {company.domain} ---")
        print(f"Final Score: {result['final_score']:.1f}/100")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Show subscore breakdown
        print("Subscores:")
        for component, details in result['subscores'].items():
            score = details['value']
            contribution = details['weighted_contribution']
            print(f"  {component.replace('_', ' ').title()}: {score:.2f} (weighted: {contribution:.2f})")
        
        # Show explanation
        print(f"Explanation: {result['explanation']}")
        
        # Show warnings if any
        if result['warnings']:
            print("Warnings:")
            for warning in result['warnings']:
                print(f"  ⚠️  {warning}")
        
        print()
    
    # Summary ranking
    print("4. Company Rankings:")
    sorted_results = sorted(results, key=lambda x: x[1]['final_score'], reverse=True)
    
    for i, (company, result) in enumerate(sorted_results, 1):
        print(f"   #{i}: {company.domain} - {result['final_score']:.1f}/100 "
              f"(confidence: {result['confidence']:.1%})")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()