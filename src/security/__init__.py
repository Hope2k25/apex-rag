"""
Security module for Apex RAG.

Provides security wrappers and mitigations for known vulnerabilities.
"""

from .pdf_security import (
    # Core types
    TrustLevel,
    SecurityConfig,
    SecurityError,
    ProcessingTimeoutError,
    
    # Decorators
    secure_pdf_processor,
    trusted_pdf_processor,
    untrusted_pdf_processor,
    
    # Utilities
    is_network_path,
    validate_file_path,
    compute_file_hash,
    isolated_processing_environment,
    with_timeout,
    with_recursion_limit,
    
    # Status
    SECURITY_STATUS,
)

__all__ = [
    "TrustLevel",
    "SecurityConfig", 
    "SecurityError",
    "ProcessingTimeoutError",
    "secure_pdf_processor",
    "trusted_pdf_processor",
    "untrusted_pdf_processor",
    "is_network_path",
    "validate_file_path",
    "compute_file_hash",
    "isolated_processing_environment",
    "with_timeout",
    "with_recursion_limit",
    "SECURITY_STATUS",
]
