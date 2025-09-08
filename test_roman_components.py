#!/usr/bin/env python3
"""
Standalone tests for Roman integration components.
"""

import sys
import json
from pathlib import Path

# Add src paths for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "roman_integration" / "adapters"))
sys.path.append(str(Path(__file__).parent / "src" / "roman_integration"))

from roman_adapter import to_company_schema
from blending import blend_scores

# Real Roman JSONL sample
SAMPLE_PLACE_NORM = {
    "entity_id": "a111c9c9-d037-4a8d-aeb7-65d8dd925a44",
    "name": "Princeton Club Xpress Monona",
    "category": "business",
    "address": {
        "line1": "5413 Monona Dr",
        "city": "WI 53716",
        "region": "USA",
        "country": "US",
        "formatted": "5413 Monona Dr, Monona, WI 53716, USA"
    },
    "website": "https://www.princetonclub.net/xpress-monona",
    "metadata": {
        "domain": "princetonclub.net",
        "rating": 4.3,
        "types": ["establishment", "gym", "health"]
    },
    "confidence_score": 0.9999999999999999,
    "created_at": "2025-09-03T23:13:19.679721+00:00"
}

SAMPLE_WEB_SNAPSHOT = {
    "pages": [{
        "text": "Join for $1 customer portal member login dashboard checkout purchase payment stripe 24-Month Plan $19.99 Bi-weekly SELECT"
    }]
}

SAMPLE_AI_SCORE = {
    "model": "gpt-4o",
    "dimb_scores": {
        "D": {"value": 0.6, "confidence": 0.8, "evidence": "Website with membership options"},
        "O": {"value": 0.4, "confidence": 0.8, "evidence": "Multiple locations, fitness services"},
        "I": {"value": 0.3, "confidence": 0.8, "evidence": "Limited evidence of complex systems"},
        "M": {"value": 0.5, "confidence": 0.8, "evidence": "Competitive fitness market"},
        "B": {"value": 0.5, "confidence": 0.8, "evidence": "Mid-tier pricing with premium features"}
    },
    "overall_score": 0.465
}

def test_adapter_functionality():
    """Test Roman adapter with real data."""
    print("=== Testing Roman Adapter ===")
    
    # Test 1: Basic conversion
    print("Test 1: Basic PlaceNorm conversion")
    company, warnings = to_company_schema(SAMPLE_PLACE_NORM)
    
    assert company.company_id == "a111c9c9-d037-4a8d-aeb7-65d8dd925a44", "Company ID mismatch"
    assert company.domain == "princetonclub.net", "Domain extraction failed"
    assert len(warnings) > 5, "Expected multiple warnings for missing data"
    print("✓ Basic conversion passed")
    
    # Test 2: Web snapshot integration
    print("Test 2: Web snapshot CRM/ecom detection")
    record_with_web = {
        "place_norm": SAMPLE_PLACE_NORM,
        "web_snapshot": SAMPLE_WEB_SNAPSHOT
    }
    
    company_web, warnings_web = to_company_schema(record_with_web)
    assert company_web.digital.crm_flag == True, "Should detect CRM markers"
    assert company_web.digital.ecom_flag == True, "Should detect ecommerce markers"
    print("✓ Web snapshot detection passed")
    
    # Test 3: Conservative defaults
    print("Test 3: Conservative extraction defaults")
    assert company.digital.pagespeed == 50, "Expected neutral pagespeed default"
    assert company.ops.employees == 1, "Expected minimal employee default"
    assert company.budget.revenue_est_usd == 100000.0, "Expected minimal revenue default"
    print("✓ Conservative defaults passed")
    
    print(f"Adapter tests completed. Generated {len(warnings)} warnings as expected.\n")
    return company

def test_blending_functionality():
    """Test score blending with different strategies."""
    print("=== Testing Score Blending ===")
    
    # Mock deterministic score
    det_score = {
        "final_score": 72.5,  # 0.725 on 0-1 scale
        "subscores": {
            "digital": {"score": 80.0, "confidence": 1.0},
            "ops": {"score": 65.0, "confidence": 1.0},
            "info_flow": {"score": 70.0, "confidence": 1.0},
            "market": {"score": 75.0, "confidence": 1.0},
            "budget": {"score": 68.0, "confidence": 1.0}
        },
        "company_id": "test-company",
        "metadata": {"version": {"engine": "1.1.0"}}
    }
    
    # Test 1: Weighted average (default 50/50)
    print("Test 1: Weighted average blending")
    result1 = blend_scores(SAMPLE_AI_SCORE, det_score)
    expected_overall = (0.465 * 0.5) + (0.725 * 0.5)  # 0.595
    assert abs(result1["blended_score"]["overall_score"] - expected_overall) < 0.001, "Weighted average calculation incorrect"
    assert result1["strategy"] == "weighted_average", "Strategy mismatch"
    print("✓ Weighted average passed")
    
    # Test 2: Max confidence strategy
    print("Test 2: Max confidence blending")
    result2 = blend_scores(SAMPLE_AI_SCORE, det_score, strategy="max_confidence")
    # Deterministic should win (confidence 1.0 vs AI ~0.8)
    assert result2["blended_score"]["overall_score"] == 0.725, "Max confidence should select deterministic"
    print("✓ Max confidence passed")
    
    # Test 3: Custom weights
    print("Test 3: Custom AI weight")
    result3 = blend_scores(SAMPLE_AI_SCORE, det_score, ai_weight=0.3)
    expected_custom = (0.465 * 0.3) + (0.725 * 0.7)  # 0.6475
    assert abs(result3["blended_score"]["overall_score"] - expected_custom) < 0.001, "Custom weight calculation incorrect"
    print("✓ Custom weights passed")
    
    # Test 4: Divergence detection
    print("Test 4: Divergence detection")
    # Create divergent scores
    divergent_ai = SAMPLE_AI_SCORE.copy()
    divergent_ai["dimb_scores"]["D"]["value"] = 0.1  # Very different from det 0.8
    
    result4 = blend_scores(divergent_ai, det_score, divergence_threshold=0.3)
    assert result4["divergence"]["has_significant_divergence"] == True, "Should detect divergence"
    assert len(result4["divergence"]["flags"]) > 0, "Should have divergence flags"
    print("✓ Divergence detection passed")
    
    print("Blending tests completed successfully.\n")
    return result1

def test_integration_workflow():
    """Test complete integration workflow."""
    print("=== Testing Integration Workflow ===")
    
    # Step 1: Roman data → CompanySchema
    record = {
        "place_norm": SAMPLE_PLACE_NORM,
        "web_snapshot": SAMPLE_WEB_SNAPSHOT
    }
    
    company, warnings = to_company_schema(record)
    print(f"Step 1: Converted {company.company_id} with {len(warnings)} warnings")
    
    # Step 2: Mock deterministic scoring result
    mock_det_result = {
        "final_score": 68.5,
        "subscores": {
            "digital": {"score": 75.0, "confidence": 1.0},
            "ops": {"score": 60.0, "confidence": 1.0}
        },
        "explanation": "Deterministic score based on extracted metrics",
        "company_id": company.company_id
    }
    print(f"Step 2: Generated deterministic score: {mock_det_result['final_score']}")
    
    # Step 3: Blend with Roman's AI score
    blended = blend_scores(SAMPLE_AI_SCORE, mock_det_result)
    print(f"Step 3: Blended result: {blended['blended_score']['overall_score']:.3f}")
    
    # Step 4: Generate final output
    integration_result = {
        "company": {
            "id": company.company_id,
            "domain": company.domain
        },
        "conversion": {
            "warnings_count": len(warnings),
            "crm_detected": company.digital.crm_flag,
            "ecom_detected": company.digital.ecom_flag
        },
        "scoring": {
            "ai_score": SAMPLE_AI_SCORE["overall_score"],
            "deterministic_score": mock_det_result["final_score"] / 100.0,
            "blended_score": blended["blended_score"]["overall_score"],
            "blending_strategy": blended["strategy"]
        },
        "quality": {
            "divergence_detected": blended["divergence"]["has_significant_divergence"],
            "divergence_flags": len(blended["divergence"]["flags"]),
            "overall_confidence": blended["blended_score"]["confidence"]
        }
    }
    
    print("✓ Integration workflow completed")
    print(f"Final result summary: {json.dumps(integration_result, indent=2)}")
    
    return integration_result

if __name__ == "__main__":
    print("Echo Ridge - Roman Integration Component Tests")
    print("==============================================")
    
    try:
        # Run component tests
        company = test_adapter_functionality()
        blend_result = test_blending_functionality()
        final_result = test_integration_workflow()
        
        print("\n🎉 All component tests passed!")
        print("\nIntegration Summary:")
        print(f"  • Company ID: {final_result['company']['id']}")
        print(f"  • Domain: {final_result['company']['domain']}")
        print(f"  • Conversion warnings: {final_result['conversion']['warnings_count']}")
        print(f"  • CRM detected: {final_result['conversion']['crm_detected']}")
        print(f"  • E-commerce detected: {final_result['conversion']['ecom_detected']}")
        print(f"  • AI score: {final_result['scoring']['ai_score']:.3f}")
        print(f"  • Deterministic score: {final_result['scoring']['deterministic_score']:.3f}")
        print(f"  • Blended score: {final_result['scoring']['blended_score']:.3f}")
        print(f"  • Strategy: {final_result['scoring']['blending_strategy']}")
        print(f"  • Divergence detected: {final_result['quality']['divergence_detected']}")
        
        print(f"\n✅ Roman integration ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)