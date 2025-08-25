#!/usr/bin/env python3
"""
Simple SDK test to isolate the datetime serialization issue
"""

import json
from src.sdk import create_client

# Test basic API call
client = create_client(base_url="http://127.0.0.1:8001")

# Simple test company data 
company_data = {
    "company_id": "test-001",
    "domain": "test.com", 
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

print("Testing direct API call...")
try:
    # Make direct API call to isolate the issue
    response = client.client.post("/score", json=company_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Score: {result['final_score']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()