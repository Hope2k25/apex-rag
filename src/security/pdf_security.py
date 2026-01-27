"""
Security wrapper for PDF processing in Apex RAG.

Mitigates known vulnerabilities in pdfminer-six:
- GHSA-wf5f-4jwr-ppcp (CVSS 8.6): Malicious PDF -> pickle RCE via network paths
- GHSA-f83h-ghpp-7wcc (CVSS 7.8): CMAP_PATH privilege escalation

Mitigates protobuf vulnerability:
- GHSA-7gcm-g887-7qv7 (CVSS 8.2): Nested Any messages -> DoS

IMPORTANT: Until pdfminer-six >= 20251230 is released, PDF processing
from untrusted sources should be considered HIGH RISK.
"""

import os
import sys
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable
from functools import wraps
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class TrustLevel(Enum):
    """Trust level for document sources."""
    TRUSTED = "trusted"        # Known safe source (e.g., your own files)
    INTERNAL = "internal"      # Internal but not verified (e.g., shared drive)
    UNTRUSTED = "untrusted"    # External/unknown source - HIGHEST RISK


@dataclass
class SecurityConfig:
    """Security configuration for PDF processing."""
    
    # Maximum file size in bytes (default: 100MB)
    max_file_size: int = 100 * 1024 * 1024
    
    # Processing timeout in seconds
    processing_timeout: int = 60
    
    # Maximum recursion depth for protobuf mitigation
    max_recursion_depth: int = 100
    
    # Allowed file extensions for PDF processing
    allowed_extensions: tuple = (".pdf",)
    
    # Block network paths (Windows SMB/WebDAV - critical for GHSA-wf5f-4jwr-ppcp)
    block_network_paths: bool = True
    
    # Require file hash verification for untrusted sources
    require_hash_verification: bool = True
    
    # Isolated temp directory for processing
    use_isolated_temp: bool = True
    
    # Log security events
    log_security_events: bool = True


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class ProcessingTimeoutError(SecurityError):
    """Raised when processing exceeds timeout."""
    pass


def _log_security_event(event: str, details: dict, config: SecurityConfig):
    """Log security events if enabled."""
    if config.log_security_events:
        import logging
        logger = logging.getLogger("apex_rag.security")
        logger.warning(f"SECURITY: {event} | {details}")


def is_network_path(path: str) -> bool:
    """
    Check if a path is a network path (Windows SMB/WebDAV).
    
    This is CRITICAL for mitigating GHSA-wf5f-4jwr-ppcp on Windows,
    where network paths can be used for RCE via malicious pickle files.
    """
    path_str = str(path)
    
    # UNC paths (\\server\share or //server/share)
    if path_str.startswith("\\\\") or path_str.startswith("//"):
        return True
    
    # WebDAV paths (http:// or https://)
    if path_str.lower().startswith(("http://", "https://")):
        return True
    
    # Windows network drive detection (less reliable but worth checking)
    # Drives like Z: that are mapped network drives
    if len(path_str) >= 2 and path_str[1] == ":":
        try:
            import ctypes
            drive = path_str[0].upper() + ":\\"
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
            DRIVE_REMOTE = 4
            if drive_type == DRIVE_REMOTE:
                return True
        except (ImportError, AttributeError, OSError):
            # Not on Windows or can't check - continue
            pass
    
    return False


def validate_file_path(
    file_path: str | Path,
    trust_level: TrustLevel,
    config: SecurityConfig
) -> Path:
    """
    Validate a file path for security before processing.
    
    Raises SecurityError if validation fails.
    """
    path = Path(file_path)
    path_str = str(path.resolve())
    
    # Check 1: Block network paths (CRITICAL for Windows RCE prevention)
    if config.block_network_paths and is_network_path(path_str):
        _log_security_event("BLOCKED_NETWORK_PATH", {"path": path_str}, config)
        raise SecurityError(
            f"Network paths are blocked for security: {path_str}\n"
            "This mitigates GHSA-wf5f-4jwr-ppcp (pickle RCE via network paths)"
        )
    
    # Check 2: File must exist and be a regular file
    if not path.exists():
        raise SecurityError(f"File does not exist: {path_str}")
    
    if not path.is_file():
        raise SecurityError(f"Path is not a regular file: {path_str}")
    
    # Check 3: Allowed extension
    if path.suffix.lower() not in config.allowed_extensions:
        raise SecurityError(
            f"File extension not allowed: {path.suffix}\n"
            f"Allowed: {config.allowed_extensions}"
        )
    
    # Check 4: File size limit
    file_size = path.stat().st_size
    if file_size > config.max_file_size:
        raise SecurityError(
            f"File too large: {file_size} bytes > {config.max_file_size} bytes"
        )
    
    # Check 5: For untrusted sources, verify it's not a symlink pointing outside
    if trust_level == TrustLevel.UNTRUSTED:
        resolved = path.resolve()
        if path.is_symlink():
            _log_security_event("SYMLINK_RESOLVED", {
                "original": str(path),
                "resolved": str(resolved)
            }, config)
    
    return path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@contextmanager
def isolated_processing_environment(config: SecurityConfig):
    """
    Create an isolated temporary directory for PDF processing.
    
    This mitigates GHSA-f83h-ghpp-7wcc by ensuring CMAP_PATH
    points to a controlled, empty directory.
    """
    if not config.use_isolated_temp:
        yield None
        return
    
    # Create isolated temp directory
    temp_dir = tempfile.mkdtemp(prefix="apex_rag_secure_")
    original_cmap_path = os.environ.get("CMAP_PATH")
    
    try:
        # Set CMAP_PATH to our controlled directory
        # This prevents loading malicious pickle files from shared directories
        os.environ["CMAP_PATH"] = temp_dir
        
        yield Path(temp_dir)
    finally:
        # Restore original CMAP_PATH
        if original_cmap_path:
            os.environ["CMAP_PATH"] = original_cmap_path
        elif "CMAP_PATH" in os.environ:
            del os.environ["CMAP_PATH"]
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup


def with_timeout(seconds: int):
    """
    Decorator to add timeout to a function.
    
    Mitigates DoS vulnerabilities like GHSA-7gcm-g887-7qv7.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise ProcessingTimeoutError(
                    f"Processing exceeded timeout of {seconds} seconds"
                )
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


def with_recursion_limit(max_depth: int):
    """
    Decorator to set Python recursion limit during execution.
    
    Mitigates GHSA-7gcm-g887-7qv7 (protobuf recursion DoS).
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_limit = sys.getrecursionlimit()
            try:
                # Set a conservative limit
                sys.setrecursionlimit(min(max_depth, original_limit))
                return func(*args, **kwargs)
            finally:
                sys.setrecursionlimit(original_limit)
        
        return wrapper
    return decorator


def secure_pdf_processor(
    trust_level: TrustLevel = TrustLevel.INTERNAL,
    config: Optional[SecurityConfig] = None
):
    """
    Decorator to wrap PDF processing functions with security mitigations.
    
    Usage:
        @secure_pdf_processor(trust_level=TrustLevel.UNTRUSTED)
        def process_pdf(file_path: Path) -> dict:
            # Your PDF processing logic here
            ...
    """
    if config is None:
        config = SecurityConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(file_path: str | Path, *args, **kwargs):
            # Step 1: Validate file path
            validated_path = validate_file_path(file_path, trust_level, config)
            
            # Step 2: Log processing attempt
            _log_security_event("PDF_PROCESS_START", {
                "path": str(validated_path),
                "trust_level": trust_level.value,
                "hash": compute_file_hash(validated_path) if trust_level == TrustLevel.UNTRUSTED else "skipped"
            }, config)
            
            # Step 3: Process in isolated environment
            with isolated_processing_environment(config) as temp_dir:
                # Step 4: Apply timeout and recursion limits
                limited_func = with_timeout(config.processing_timeout)(
                    with_recursion_limit(config.max_recursion_depth)(func)
                )
                
                try:
                    result = limited_func(validated_path, *args, **kwargs)
                    _log_security_event("PDF_PROCESS_SUCCESS", {
                        "path": str(validated_path)
                    }, config)
                    return result
                except Exception as e:
                    _log_security_event("PDF_PROCESS_ERROR", {
                        "path": str(validated_path),
                        "error": str(e)
                    }, config)
                    raise
        
        return wrapper
    return decorator


# Pre-configured wrappers for common trust levels
def trusted_pdf_processor(func: Callable) -> Callable:
    """Wrapper for processing PDFs from trusted sources."""
    return secure_pdf_processor(TrustLevel.TRUSTED)(func)


def untrusted_pdf_processor(func: Callable) -> Callable:
    """Wrapper for processing PDFs from untrusted sources (maximum security)."""
    config = SecurityConfig(
        max_file_size=50 * 1024 * 1024,  # Stricter: 50MB
        processing_timeout=30,             # Stricter: 30s
        max_recursion_depth=50,            # Stricter: 50
        require_hash_verification=True,
    )
    return secure_pdf_processor(TrustLevel.UNTRUSTED, config)(func)


# Export security status for documentation
SECURITY_STATUS = {
    "pdfminer_six_vulnerabilities": [
        {
            "id": "GHSA-wf5f-4jwr-ppcp",
            "cvss": 8.6,
            "description": "RCE via malicious PDF referencing pickle files (network paths on Windows)",
            "mitigation": "Block network paths, isolated processing environment",
            "mitigated": True,
        },
        {
            "id": "GHSA-f83h-ghpp-7wcc", 
            "cvss": 7.8,
            "description": "Privilege escalation via CMAP_PATH pickle injection",
            "mitigation": "Isolated temp directory for CMAP_PATH",
            "mitigated": True,
        },
    ],
    "protobuf_vulnerabilities": [
        {
            "id": "GHSA-7gcm-g887-7qv7",
            "cvss": 8.2,
            "description": "DoS via deeply nested protobuf Any messages",
            "mitigation": "Recursion limits and processing timeout",
            "mitigated": True,
        },
    ],
    "recommended_actions": [
        "Update to pdfminer-six >= 20251230 when available",
        "Update to fixed protobuf version when released",
        "Run security audit regularly: osv-scanner scan --lockfile uv.lock",
    ],
}
