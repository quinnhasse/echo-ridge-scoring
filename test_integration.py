#!/usr/bin/env python3
"""
Integration test for Roman adapter and blending functionality.
"""

import sys
import json
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    sys.path.append(str(Path(__file__).parent / "src" / "roman_integration" / "adapters"))
    sys.path.append(str(Path(__file__).parent / "src" / "roman_integration"))
    
    from roman_adapter import to_company_schema
    from blending import blend_scores
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test data from Roman's system
ROMAN_SAMPLE = {
    "entity_id": "test-gym-001",
    "name": "Test Fitness Center", 
    "website": "https://testgym.com",
    "metadata": {
        "domain": "testgym.com",
        "rating": 4.2,
        "types": ["gym", "health"]
    },
    "confidence_score": 0.9,
    "created_at": "2025-09-08T12:00:00+00:00"
}

SAMPLE_WEB_SNAPSHOT = {
    "pages": [{
        "text": "Join now! Member portal login available. Online payment checkout with credit card processing. Personal training services."
    }]
}

AI_SCORE_SAMPLE = {
    "model": "gpt-4o",
    "dimb_scores": {
        "D": {"value": 0.7, "confidence": 0.8, "evidence": "Good website"},
        "O": {"value": 0.5, "confidence": 0.8, "evidence": "Medium operations"},
        "I": {"value": 0.4, "confidence": 0.7, "evidence": "Basic info systems"},
        "M": {"value": 0.6, "confidence": 0.8, "evidence": "Competitive market"},
        "B": {"value": 0.5, "confidence": 0.8, "evidence": "Standard pricing"}
    },
    "overall_score": 0.54
}

def test_adapter():
    """Test Roman adapter conversion."""
    print("\n=== Testing Roman Adapter ===")
    
    # Test basic conversion
    company, warnings = to_company_schema(ROMAN_SAMPLE)
    
    print(f"✓ Converted company ID: {company.company_id}")
    print(f"✓ Extracted domain: {company.domain}")
    print(f"✓ Warnings generated: {len(warnings)}")
    
    # Test with web snapshot
    record_with_web = {
        "place_norm": ROMAN_SAMPLE,
        "web_snapshot": SAMPLE_WEB_SNAPSHOT
    }
    
    company_web, warnings_web = to_company_schema(record_with_web)
    
    print(f"✓ CRM detection: {company_web.digital.crm_flag}")
    print(f"✓ E-commerce detection: {company_web.digital.ecom_flag}")
    
    return company

def test_blending():
    """Test score blending functionality."""
    print("\n=== Testing Score Blending ===")
    
    # Mock deterministic score
    det_score = {
        "final_score": 65.0,
        "subscores": {
            "digital": {"score": 70.0, "confidence": 1.0},
            "ops": {"score": 60.0, "confidence": 1.0},
            "info_flow": {"score": 55.0, "confidence": 1.0},
            "market": {"score": 75.0, "confidence": 1.0},
            "budget": {"score": 65.0, "confidence": 1.0}
        },
        "explanation": "Deterministic calculation",
        "company_id": "test-gym-001",
        "metadata": {"version": {"engine": "1.1.0"}}
    }
    
    # Test weighted average (default)
    result1 = blend_scores(AI_SCORE_SAMPLE, det_score)
    print(f"✓ Weighted average: {result1['blended_score']['overall_score']:.3f}")
    
    # Test max confidence
    result2 = blend_scores(AI_SCORE_SAMPLE, det_score, strategy="max_confidence")
    print(f"✓ Max confidence: {result2['blended_score']['overall_score']:.3f}")
    
    # Test consensus
    result3 = blend_scores(AI_SCORE_SAMPLE, det_score, strategy="consensus")
    print(f"✓ Consensus: {result3['blended_score']['overall_score']:.3f}")
    
    # Test divergence detection
    divergence_flags = result1['divergence']['flags']
    print(f"✓ Divergence flags: {len(divergence_flags)}")
    
    return result1

def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\n=== Testing End-to-End Integration ===")
    
    # Step 1: Convert Roman data
    record = {
        "place_norm": ROMAN_SAMPLE,
        "web_snapshot": SAMPLE_WEB_SNAPSHOT
    }
    
    company, warnings = to_company_schema(record)
    print(f"✓ Step 1 - Conversion: {company.company_id}")
    
    # Step 2: Mock deterministic scoring (would normally call score_company)
    mock_det_score = {
        "final_score": 68.5,
        "subscores": {"digital": {"score": 75.0, "confidence": 1.0}},
        "company_id": company.company_id,
        "explanation": "Mock deterministic score"
    }
    print(f"✓ Step 2 - Deterministic scoring: {mock_det_score['final_score']}")
    
    # Step 3: Blend with AI score
    blended = blend_scores(AI_SCORE_SAMPLE, mock_det_score)
    print(f"✓ Step 3 - Blending: {blended['blended_score']['overall_score']:.3f}")
    
    # Step 4: Output results
    final_result = {
        "company_id": company.company_id,
        "conversion_warnings": len(warnings),
        "ai_score": AI_SCORE_SAMPLE["overall_score"],
        "deterministic_score": mock_det_score["final_score"] / 100.0,
        "blended_score": blended['blended_score']['overall_score'],
        "divergence_detected": blended['divergence']['has_significant_divergence'],
        "blending_strategy": blended['strategy']
    }
    
    print(f"✓ Final result: {json.dumps(final_result, indent=2)}")
    return final_result

if __name__ == "__main__":
    print("Echo Ridge - Roman Integration Test")
    print("===================================")
    
    try:
        # Run tests
        company = test_adapter()
        blend_result = test_blending() 
        final_result = test_end_to_end()
        
        print(f"\n🎉 All tests passed!")
        print(f"   - Adapter warnings: {final_result['conversion_warnings']}")
        print(f"   - Blended score: {final_result['blended_score']:.3f}")
        print(f"   - Strategy used: {final_result['blending_strategy']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)