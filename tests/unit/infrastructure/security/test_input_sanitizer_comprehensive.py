"""
Comprehensive unit tests for Input Sanitizer.

Tests the input sanitization system including SQL injection prevention,
XSS prevention, filename sanitization, and URL sanitization with full coverage.
"""

# Standard library imports
import html
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from src.infrastructure.security.input_sanitizer import InputSanitizer, SanitizationError


@pytest.mark.unit
class TestInputSanitizerBasics:
    """Test basic input sanitizer functionality."""

    def test_sanitization_error_exception(self):
        """Test SanitizationError exception."""
        error = SanitizationError("Invalid input")
        assert str(error) == "Invalid input"

    def test_sql_injection_patterns_removed(self):
        """Test that SQL injection patterns have been removed for security."""
        # SQL_INJECTION_PATTERNS should no longer exist
        assert not hasattr(
            InputSanitizer, "SQL_INJECTION_PATTERNS"
        ), "SQL_INJECTION_PATTERNS should be removed - SQL injection prevention should use parameterized queries"

    def test_xss_patterns(self):
        """Test XSS pattern definitions."""
        patterns = InputSanitizer.XSS_PATTERNS

        assert any("script" in p for p in patterns)
        assert any("javascript:" in p for p in patterns)
        assert any("on\\w+" in p for p in patterns)


@pytest.mark.unit
class TestSanitizeString:
    """Test string sanitization."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        result = InputSanitizer.sanitize_string("Hello World")
        assert result == "Hello World"

    def test_sanitize_string_with_whitespace(self):
        """Test string sanitization with whitespace."""
        result = InputSanitizer.sanitize_string("  Hello World  ")
        assert result == "Hello World"

    def test_sanitize_string_html_escape(self):
        """Test HTML escaping in string sanitization."""
        result = InputSanitizer.sanitize_string("<div>Hello</div>")
        assert result == html.escape("<div>Hello</div>")
        assert "&lt;" in result and "&gt;" in result

    def test_sanitize_string_non_string_input(self):
        """Test sanitizing non-string input."""
        result = InputSanitizer.sanitize_string(123)
        assert result == "123"

        result = InputSanitizer.sanitize_string(True)
        assert result == "True"

        result = InputSanitizer.sanitize_string(None)
        assert result == "None"

    def test_sanitize_string_max_length(self):
        """Test string truncation with max_length."""
        long_string = "a" * 100
        result = InputSanitizer.sanitize_string(long_string, max_length=50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_sanitize_string_sql_injection_detected(self):
        """Test detection of SQL injection patterns."""
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "admin' --",
            "SELECT * FROM users",
            "'; INSERT INTO admin VALUES ('hacker', 'password'); --",
            "UNION SELECT password FROM users",
            "'; EXEC sp_addlogin 'hacker', 'password'; --",
        ]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(SanitizationError, match="potentially dangerous patterns"):
                InputSanitizer.sanitize_string(dangerous_input)

    def test_sanitize_string_xss_detected(self):
        """Test detection of XSS patterns."""
        dangerous_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<a href='javascript:void(0)'>Click</a>",
            "<div onclick='alert(1)'>Test</div>",
        ]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(SanitizationError, match="potentially dangerous patterns"):
                InputSanitizer.sanitize_string(dangerous_input)

    def test_sanitize_string_case_insensitive_detection(self):
        """Test case-insensitive pattern detection."""
        dangerous_inputs = [
            "SELECT * from users",  # lowercase
            "select * FROM users",  # mixed case
            "SeLeCt * FrOm users",  # mixed case
            "<SCRIPT>alert('XSS')</SCRIPT>",  # uppercase
            "JavaScript:alert('XSS')",  # mixed case
        ]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(SanitizationError, match="potentially dangerous patterns"):
                InputSanitizer.sanitize_string(dangerous_input)

    def test_sanitize_string_safe_inputs(self):
        """Test that safe inputs pass through."""
        safe_inputs = [
            "John Doe",
            "user@example.com",
            "123-456-7890",
            "Product description with special chars: $100.50",
            "This is a normal sentence.",
            "2024-01-01",
        ]

        for safe_input in safe_inputs:
            result = InputSanitizer.sanitize_string(safe_input)
            assert html.escape(safe_input) == result


@pytest.mark.unit
class TestContainsDangerousPattern:
    """Test dangerous pattern detection."""

    def test_contains_dangerous_pattern_sql(self):
        """Test SQL injection pattern detection."""
        assert InputSanitizer._contains_dangerous_pattern("SELECT * FROM users") is True
        assert InputSanitizer._contains_dangerous_pattern("DROP TABLE users") is True
        assert InputSanitizer._contains_dangerous_pattern("1 OR 1=1") is True
        assert InputSanitizer._contains_dangerous_pattern("1 AND 1=1") is True
        assert InputSanitizer._contains_dangerous_pattern("--comment") is True
        assert InputSanitizer._contains_dangerous_pattern("/* comment */") is True

    def test_contains_dangerous_pattern_xss(self):
        """Test XSS pattern detection."""
        assert InputSanitizer._contains_dangerous_pattern("<script>alert(1)</script>") is True
        assert InputSanitizer._contains_dangerous_pattern("javascript:void(0)") is True
        assert InputSanitizer._contains_dangerous_pattern("onclick='alert(1)'") is True
        assert InputSanitizer._contains_dangerous_pattern("onmouseover=alert(1)") is True

    def test_contains_dangerous_pattern_safe(self):
        """Test that safe patterns are not flagged."""
        assert InputSanitizer._contains_dangerous_pattern("Hello World") is False
        assert InputSanitizer._contains_dangerous_pattern("user@example.com") is False
        assert InputSanitizer._contains_dangerous_pattern("123456") is False
        assert InputSanitizer._contains_dangerous_pattern("Product Name") is False


@pytest.mark.unit
class TestSanitizeSQLIdentifier:
    """Test SQL identifier sanitization."""

    def test_sanitize_sql_identifier_valid(self):
        """Test valid SQL identifiers."""
        valid_identifiers = [
            "users",
            "user_id",
            "firstName",
            "table123",
            "my_table_name",
        ]

        for identifier in valid_identifiers:
            result = InputSanitizer.sanitize_sql_identifier(identifier)
            assert result == identifier

    def test_sanitize_sql_identifier_invalid(self):
        """Test invalid SQL identifiers."""
        invalid_identifiers = [
            "123table",  # Starts with number
            "table-name",  # Contains hyphen
            "table.name",  # Contains dot
            "table name",  # Contains space
            "table@name",  # Contains special char
            "_table",  # Starts with underscore
            "",  # Empty string
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(SanitizationError, match="Invalid SQL identifier"):
                InputSanitizer.sanitize_sql_identifier(identifier)

    def test_sanitize_sql_identifier_reserved_words(self):
        """Test that reserved SQL words are rejected."""
        reserved_words = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "select",  # lowercase
            "Select",  # mixed case
        ]

        for word in reserved_words:
            with pytest.raises(SanitizationError, match="Reserved SQL word"):
                InputSanitizer.sanitize_sql_identifier(word)


@pytest.mark.unit
class TestSqlValueSanitizationRemoved:
    """Test that dangerous SQL value sanitization has been removed."""

    def test_sanitize_sql_value_method_removed(self):
        """Test that sanitize_sql_value method has been removed for security."""
        # Method should no longer exist
        assert not hasattr(
            InputSanitizer, "sanitize_sql_value"
        ), "sanitize_sql_value should be removed - use parameterized queries instead"

    def test_replacement_method_guidance(self):
        """Test that replacement method provides proper guidance."""
        with pytest.raises(NotImplementedError) as exc_info:
            InputSanitizer._removed_sanitize_sql_value("test")

        error_msg = str(exc_info.value).lower()
        assert "parameterized queries" in error_msg
        assert "removed for security" in error_msg
        assert "cursor.execute" in error_msg  # Should show example

    def test_security_documentation_present(self):
        """Test that security warnings are present in code."""
        import inspect

        source = inspect.getsource(InputSanitizer)

        # Should contain security warnings about parameterized queries
        assert "parameterized queries" in source.lower()
        assert "security" in source.lower()
        # Should warn against manual SQL escaping
        assert "dangerous" in source.lower() or "removed" in source.lower()


@pytest.mark.unit
class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = InputSanitizer.sanitize_filename("document.pdf")
        assert result == "document.pdf"

        result = InputSanitizer.sanitize_filename("my-file_123.txt")
        assert result == "my-file_123.txt"

    def test_sanitize_filename_directory_traversal(self):
        """Test removal of directory traversal attempts."""
        result = InputSanitizer.sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

        result = InputSanitizer.sanitize_filename("..\\..\\windows\\system32")
        assert ".." not in result
        assert "\\" not in result

    def test_sanitize_filename_special_characters(self):
        """Test replacement of special characters."""
        result = InputSanitizer.sanitize_filename("file<>:|?*.txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_sanitize_filename_spaces_and_unicode(self):
        """Test handling of spaces and unicode characters."""
        result = InputSanitizer.sanitize_filename("my file name.txt")
        assert result == "my_file_name.txt"  # Spaces replaced

        result = InputSanitizer.sanitize_filename("ñoño.txt")
        assert "ñ" not in result  # Non-ASCII removed

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting."""
        long_name = "a" * 300 + ".txt"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".txt")

        # Test without extension
        long_name_no_ext = "b" * 300
        result = InputSanitizer.sanitize_filename(long_name_no_ext)
        assert len(result) <= 255

    def test_sanitize_filename_preserve_extension(self):
        """Test that file extension is preserved when truncating."""
        long_name = "x" * 260 + ".important.extension"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".extension")

    def test_sanitize_filename_empty_and_dots(self):
        """Test edge cases with empty names and dots."""
        result = InputSanitizer.sanitize_filename("...")
        assert result == "..."

        result = InputSanitizer.sanitize_filename(".hidden")
        assert result == ".hidden"


@pytest.mark.unit
class TestSanitizeURL:
    """Test URL sanitization."""

    def test_sanitize_url_basic(self):
        """Test basic URL sanitization."""
        result = InputSanitizer.sanitize_url("https://example.com")
        assert result == "https://example.com"

        result = InputSanitizer.sanitize_url("http://example.com/path")
        assert result == "http://example.com/path"

    def test_sanitize_url_javascript_scheme(self):
        """Test rejection of javascript: URLs."""
        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("javascript:alert('XSS')")

        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("JavaScript:void(0)")

    def test_sanitize_url_data_scheme(self):
        """Test rejection of data: URLs."""
        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("data:text/html,<script>alert('XSS')</script>")

        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("DATA:text/html,content")

    def test_sanitize_url_vbscript_scheme(self):
        """Test rejection of vbscript: URLs."""
        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("vbscript:msgbox('XSS')")

        with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
            InputSanitizer.sanitize_url("VBScript:code")

    def test_sanitize_url_special_character_encoding(self):
        """Test encoding of special characters in URLs."""
        result = InputSanitizer.sanitize_url("https://example.com/path with spaces")
        assert "path%20with%20spaces" in result

        result = InputSanitizer.sanitize_url("https://example.com/<script>")
        assert "%3C" in result and "%3E" in result

        result = InputSanitizer.sanitize_url('https://example.com/"quotes"')
        assert "%22" in result

    def test_sanitize_url_query_parameters(self):
        """Test URL with query parameters."""
        result = InputSanitizer.sanitize_url("https://example.com?param=value&other=123")
        assert "param=value" in result
        assert "other=123" in result

        result = InputSanitizer.sanitize_url("https://example.com?search=hello world")
        assert "hello%20world" in result

    def test_sanitize_url_fragment(self):
        """Test URL with fragment."""
        result = InputSanitizer.sanitize_url("https://example.com/page#section")
        assert "#section" in result

        result = InputSanitizer.sanitize_url("https://example.com#anchor with space")
        assert "anchor%20with%20space" in result

    def test_sanitize_url_case_insensitive_scheme_check(self):
        """Test case-insensitive scheme checking."""
        dangerous_schemes = [
            "JaVaScRiPt:alert(1)",
            "DaTa:text/html,content",
            "vBsCrIpT:code",
        ]

        for url in dangerous_schemes:
            with pytest.raises(SanitizationError, match="Potentially dangerous URL scheme"):
                InputSanitizer.sanitize_url(url)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        result = InputSanitizer.sanitize_string("")
        assert result == ""

        result = InputSanitizer.sanitize_filename("")
        assert result == ""

        result = InputSanitizer.sanitize_url("")
        assert result == ""

    def test_very_long_inputs(self):
        """Test handling of very long inputs."""
        long_string = "a" * 10000
        result = InputSanitizer.sanitize_string(long_string, max_length=1000)
        assert len(result) == 1000

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        unicode_string = "Hello 世界 🌍"
        result = InputSanitizer.sanitize_string(unicode_string)
        assert "世界" in result
        assert "🌍" in result

    def test_logging_dangerous_patterns(self):
        """Test that dangerous patterns trigger logging."""
        with patch("src.infrastructure.security.input_sanitizer.logger") as mock_logger:
            try:
                InputSanitizer.sanitize_string("SELECT * FROM users WHERE id=1")
            except SanitizationError:
                pass

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Dangerous pattern detected" in call_args
