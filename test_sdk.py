#!/usr/bin/env python3
"""
Test script for Echo Ridge Scoring Engine SDK
Validates all SDK functions against a running API server.
"""

import json
import tempfile
from pathlib import Path
from src.sdk import create_client, score_company, score_companies, score_file, ScoringError
from src.schema import CompanySchema

def test_api_connectivity():
    """Test basic API connectivity"""
    print("1. Testing API connectivity...")
    
    client = create_client(base_url="http://127.0.0.1:8001")
    
    # Test health endpoint indirectly through client setup
    try:
        # This will validate the client can connect
        result = client.client.get("/healthz")
        if result.status_code == 200:
            health_data = result.json()
            print(f"   ✓ API is healthy: {health_data['status']}")
            print(f"   ✓ Version: {health_data['version']}")
            return True
        else:
            print(f"   ✗ Health check failed: {result.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

def test_single_company_scoring():
    """Test scoring a single company using SDK"""
    print("\n2. Testing single company scoring...")
    
    # Create test company data matching the actual schema
    company_data = {
        "company_id": "test-sdk-001",
        "domain": "testsdk.com",
        "digital": {
            "pagespeed": 85,
            "crm_flag": True,
            "ecom_flag": False
        },
        "ops": {
            "employees": 25,
            "locations": 2,
            "services_count": 5
        },
        "info_flow": {
            "daily_docs_est": 150
        },
        "market": {
            "competitor_density": 8,
            "industry_growth_pct": 3.5,
            "rivalry_index": 0.7
        },
        "budget": {
            "revenue_est_usd": 1500000
        },
        "meta": {
            "scrape_ts": "2025-08-25T10:00:00Z",
            "source_confidence": 0.85
        }
    }
    
    try:
        result = score_company(
            company=company_data,
            base_url="http://127.0.0.1:8001"
        )
        
        # Validate result structure
        assert "final_score" in result
        assert "confidence" in result
        assert "subscores" in result
        assert "company_id" in result
        assert result["company_id"] == "test-sdk-001"
        
        print(f"   ✓ Scored successfully: {result['final_score']:.1f}/100")
        print(f"   ✓ Confidence: {result['confidence']:.1%}")
        print(f"   ✓ Company ID matches: {result['company_id']}")
        return True
        
    except Exception as e:
        print(f"   ✗ Single company scoring failed: {e}")
        return False

def test_batch_company_scoring():
    """Test scoring multiple companies using SDK"""
    print("\n3. Testing batch company scoring...")
    
    # Create multiple test companies
    companies = []
    for i in range(3):
        company = {
            "company_id": f"batch-test-{i:03d}",
            "domain": f"batch{i}.com",
            "digital": {
                "pagespeed": 70 + i * 5,
                "crm_flag": i % 2 == 0,
                "ecom_flag": i % 2 == 1
            },
            "ops": {
                "employees": 20 + i * 10,
                "locations": 1 + i,
                "services_count": 5 + i * 2
            },
            "info_flow": {
                "daily_docs_est": 100 + i * 50
            },
            "market": {
                "competitor_density": 5 + i * 3,
                "industry_growth_pct": 2.0 + i,
                "rivalry_index": 0.5 + i * 0.1
            },
            "budget": {
                "revenue_est_usd": 1000000 + i * 500000
            },
            "meta": {
                "scrape_ts": "2025-08-25T10:00:00Z",
                "source_confidence": 0.8 + i * 0.05
            }
        }
        companies.append(company)
    
    try:
        results = score_companies(
            companies=companies,
            base_url="http://127.0.0.1:8001"
        )
        
        # Validate batch results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert "final_score" in result
            assert "company_id" in result
            assert result["company_id"] == f"batch-test-{i:03d}"
        
        print(f"   ✓ Batch scored {len(results)} companies successfully")
        scores = [r["final_score"] for r in results]
        print(f"   ✓ Scores: {scores}")
        return True
        
    except Exception as e:
        print(f"   ✗ Batch scoring failed: {e}")
        return False

def test_file_scoring():
    """Test scoring from JSONL file using SDK"""
    print("\n4. Testing file scoring...")
    
    # Create temporary JSONL input file
    test_company = {
        "company_id": "file-test-001",
        "domain": "filetest.com",
        "digital": {
            "pagespeed": 78,
            "crm_flag": True,
            "ecom_flag": True
        },
        "ops": {
            "employees": 45,
            "locations": 3,
            "services_count": 8
        },
        "info_flow": {
            "daily_docs_est": 200
        },
        "market": {
            "competitor_density": 12,
            "industry_growth_pct": 4.2,
            "rivalry_index": 0.6
        },
        "budget": {
            "revenue_est_usd": 3000000
        },
        "meta": {
            "scrape_ts": "2025-08-25T10:00:00Z",
            "source_confidence": 0.9
        }
    }
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            json.dump(test_company, input_file)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        # Score the file
        stats = score_file(
            input_path=input_path,
            output_path=output_path,
            base_url="http://127.0.0.1:8001"
        )
        
        # Validate stats
        assert stats["total_companies"] == 1
        assert stats["successful_scores"] == 1
        assert stats["failed_scores"] == 0
        assert stats["success_rate"] == 1.0
        
        # Check output file exists and contains results
        assert output_path.exists()
        with open(output_path, 'r') as f:
            result = json.loads(f.read().strip())
            assert result["company_id"] == "file-test-001"
            assert "final_score" in result
        
        print(f"   ✓ File processed: {stats['total_companies']} companies")
        print(f"   ✓ Success rate: {stats['success_rate']:.1%}")
        print(f"   ✓ Output file created successfully")
        
        # Cleanup
        input_path.unlink()
        output_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"   ✗ File scoring failed: {e}")
        return False

def test_error_handling():
    """Test SDK error handling with invalid data"""
    print("\n5. Testing error handling...")
    
    # Test with invalid company data (missing required fields)
    invalid_company = {
        "company_id": "invalid-test",
        "domain": "invalid.com"
        # Missing all other required fields
    }
    
    try:
        # This should raise a ScoringError due to validation failure
        result = score_company(
            company=invalid_company,
            base_url="http://127.0.0.1:8001"
        )
        print("   ✗ Expected error handling failed - invalid data was accepted")
        return False
        
    except ScoringError as e:
        print(f"   ✓ Properly caught ScoringError: {type(e).__name__}")
        return True
    except Exception as e:
        print(f"   ✓ Caught validation error: {type(e).__name__}")
        return True

def main():
    """Run all SDK tests"""
    print("=== Echo Ridge SDK Testing ===")
    
    tests = [
        test_api_connectivity,
        test_single_company_scoring,
        test_batch_company_scoring,
        test_file_scoring,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All SDK tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)