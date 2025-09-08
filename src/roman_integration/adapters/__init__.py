"""
Adapters for converting external data formats to Echo Ridge CompanySchema.
"""

from .roman_adapter import to_company_schema

__all__ = ["to_company_schema"]