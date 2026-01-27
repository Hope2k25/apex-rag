"""
Tests for the PDF security module.

Run with: uv run pytest tests/test_security.py -v
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from security.pdf_security import (
    TrustLevel,
    SecurityConfig,
    SecurityError,
    ProcessingTimeoutError,
    is_network_path,
    validate_file_path,
    compute_file_hash,
    isolated_processing_environment,
    with_timeout,
    with_recursion_limit,
    secure_pdf_processor,
)


class TestIsNetworkPath:
    """Tests for network path detection."""
    
    def test_unc_path_backslash(self):
        """UNC paths with backslashes should be detected."""
        assert is_network_path("\\\\server\\share\\file.pdf") is True
        
    def test_unc_path_forward_slash(self):
        """UNC paths with forward slashes should be detected."""
        assert is_network_path("//server/share/file.pdf") is True
    
    def test_http_url(self):
        """HTTP URLs should be detected as network paths."""
        assert is_network_path("http://example.com/file.pdf") is True
        
    def test_https_url(self):
        """HTTPS URLs should be detected as network paths."""
        assert is_network_path("https://example.com/file.pdf") is True
    
    def test_local_path_absolute(self):
        """Local absolute paths should NOT be detected as network."""
        assert is_network_path("C:\\Users\\test\\file.pdf") is False
        
    def test_local_path_relative(self):
        """Local relative paths should NOT be detected as network."""
        assert is_network_path("./file.pdf") is False
        assert is_network_path("../file.pdf") is False
        
    def test_local_path_unix_style(self):
        """Unix-style paths should NOT be detected as network."""
        assert is_network_path("/home/user/file.pdf") is False


class TestValidateFilePath:
    """Tests for file path validation."""
    
    @pytest.fixture
    def temp_pdf(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content")
            temp_path = f.name
        yield Path(temp_path)
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
    
    @pytest.fixture
    def config(self):
        """Default security config for tests."""
        return SecurityConfig(log_security_events=False)
    
    def test_valid_pdf_file(self, temp_pdf, config):
        """Valid PDF file should pass validation."""
        result = validate_file_path(temp_pdf, TrustLevel.INTERNAL, config)
        assert result == temp_pdf
    
    def test_network_path_blocked(self, config):
        """Network paths should be blocked."""
        with pytest.raises(SecurityError) as exc_info:
            validate_file_path("\\\\server\\share\\file.pdf", TrustLevel.TRUSTED, config)
        assert "Network paths are blocked" in str(exc_info.value)
    
    def test_nonexistent_file(self, config):
        """Nonexistent files should fail validation."""
        with pytest.raises(SecurityError) as exc_info:
            validate_file_path("/nonexistent/file.pdf", TrustLevel.TRUSTED, config)
        assert "does not exist" in str(exc_info.value)
    
    def test_wrong_extension(self, config):
        """Wrong file extensions should fail validation."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        try:
            with pytest.raises(SecurityError) as exc_info:
                validate_file_path(temp_path, TrustLevel.TRUSTED, config)
            assert "extension not allowed" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_file_too_large(self, temp_pdf):
        """Files exceeding size limit should fail validation."""
        config = SecurityConfig(max_file_size=10, log_security_events=False)  # 10 bytes
        with pytest.raises(SecurityError) as exc_info:
            validate_file_path(temp_pdf, TrustLevel.TRUSTED, config)
        assert "too large" in str(exc_info.value)


class TestComputeFileHash:
    """Tests for file hash computation."""
    
    def test_hash_consistency(self):
        """Same content should produce same hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for hashing")
            temp_path = f.name
        try:
            hash1 = compute_file_hash(Path(temp_path))
            hash2 = compute_file_hash(Path(temp_path))
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 produces 64 hex chars
        finally:
            os.unlink(temp_path)
    
    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content A")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content B")
            path2 = f2.name
        try:
            hash1 = compute_file_hash(Path(path1))
            hash2 = compute_file_hash(Path(path2))
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestIsolatedProcessingEnvironment:
    """Tests for isolated processing environment."""
    
    def test_cmap_path_is_set(self):
        """CMAP_PATH should be set to temp directory during processing."""
        config = SecurityConfig(use_isolated_temp=True)
        original_cmap = os.environ.get("CMAP_PATH")
        
        with isolated_processing_environment(config) as temp_dir:
            assert temp_dir is not None
            assert temp_dir.exists()
            assert os.environ.get("CMAP_PATH") == str(temp_dir)
        
        # Should be restored after
        assert os.environ.get("CMAP_PATH") == original_cmap
    
    def test_temp_dir_cleaned_up(self):
        """Temp directory should be cleaned up after processing."""
        config = SecurityConfig(use_isolated_temp=True)
        
        with isolated_processing_environment(config) as temp_dir:
            temp_path = temp_dir
            assert temp_path.exists()
        
        # Should be deleted after context exits
        assert not temp_path.exists()
    
    def test_disabled_isolation(self):
        """When disabled, no temp dir should be created."""
        config = SecurityConfig(use_isolated_temp=False)
        
        with isolated_processing_environment(config) as temp_dir:
            assert temp_dir is None


class TestWithTimeout:
    """Tests for timeout decorator."""
    
    def test_fast_function_completes(self):
        """Functions completing within timeout should work."""
        @with_timeout(5)
        def fast_function():
            return "completed"
        
        assert fast_function() == "completed"
    
    def test_slow_function_times_out(self):
        """Functions exceeding timeout should raise error."""
        import time
        
        @with_timeout(1)
        def slow_function():
            time.sleep(10)
            return "completed"
        
        with pytest.raises(ProcessingTimeoutError):
            slow_function()
    
    def test_exception_propagates(self):
        """Exceptions inside function should propagate."""
        @with_timeout(5)
        def error_function():
            raise ValueError("test error")
        
        with pytest.raises(ValueError) as exc_info:
            error_function()
        assert "test error" in str(exc_info.value)


class TestWithRecursionLimit:
    """Tests for recursion limit decorator."""
    
    def test_normal_function_works(self):
        """Normal functions should work within limit."""
        @with_recursion_limit(100)
        def normal_function():
            return sum(range(10))
        
        assert normal_function() == 45
    
    def test_limit_is_enforced(self):
        """Recursion exceeding limit should fail."""
        @with_recursion_limit(50)
        def deep_recursion(n):
            if n <= 0:
                return 0
            return 1 + deep_recursion(n - 1)
        
        # This should hit recursion limit
        with pytest.raises(RecursionError):
            deep_recursion(100)
    
    def test_limit_restored_after(self):
        """Original recursion limit should be restored."""
        original_limit = sys.getrecursionlimit()
        
        @with_recursion_limit(50)
        def test_func():
            return True
        
        test_func()
        assert sys.getrecursionlimit() == original_limit


class TestSecurePdfProcessor:
    """Tests for the secure PDF processor decorator."""
    
    @pytest.fixture
    def temp_pdf(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content for processing")
            temp_path = f.name
        yield Path(temp_path)
        try:
            os.unlink(temp_path)
        except:
            pass
    
    def test_processes_valid_pdf(self, temp_pdf):
        """Valid PDFs should be processed successfully."""
        config = SecurityConfig(log_security_events=False)
        
        @secure_pdf_processor(TrustLevel.TRUSTED, config)
        def process(path):
            return f"processed: {path.name}"
        
        result = process(temp_pdf)
        assert "processed:" in result
    
    def test_blocks_network_path(self):
        """Network paths should be blocked."""
        config = SecurityConfig(log_security_events=False)
        
        @secure_pdf_processor(TrustLevel.TRUSTED, config)
        def process(path):
            return "should not reach here"
        
        with pytest.raises(SecurityError) as exc_info:
            process("\\\\server\\share\\file.pdf")
        assert "Network paths are blocked" in str(exc_info.value)


class TestTrustLevels:
    """Tests for trust level enum."""
    
    def test_trust_levels_exist(self):
        """All trust levels should be defined."""
        assert TrustLevel.TRUSTED.value == "trusted"
        assert TrustLevel.INTERNAL.value == "internal"
        assert TrustLevel.UNTRUSTED.value == "untrusted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
