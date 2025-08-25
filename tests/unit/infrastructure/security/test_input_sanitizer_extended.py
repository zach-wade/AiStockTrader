"""
Extended unit tests for input sanitizer module to achieve 80%+ coverage.

Focuses on SQL injection prevention, XSS protection, and comprehensive
sanitization of all input types.
"""

from unittest.mock import patch

import pytest

from src.infrastructure.security.input_sanitizer import InputSanitizer, SanitizationError


class TestInputSanitizerStringMethods:
    """Test string sanitization methods"""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization"""
        result = InputSanitizer.sanitize_string("  hello world  ")
        assert result == "hello world"

    def test_sanitize_string_non_string_input(self):
        """Test sanitization with non-string input"""
        result = InputSanitizer.sanitize_string(123)
        assert result == "123"

        result = InputSanitizer.sanitize_string(True)
        assert result == "True"

        result = InputSanitizer.sanitize_string(None)
        assert result == "None"

    def test_sanitize_string_html_escape(self):
        """Test HTML escaping"""
        result = InputSanitizer.sanitize_string("<div>test</div>")
        assert result == "&lt;div&gt;test&lt;/div&gt;"

        result = InputSanitizer.sanitize_string("test & test")
        assert result == "test &amp; test"

        result = InputSanitizer.sanitize_string('"quoted"')
        assert result == "&quot;quoted&quot;"

    def test_sanitize_string_max_length(self):
        """Test string truncation with max_length"""
        long_string = "a" * 100
        result = InputSanitizer.sanitize_string(long_string, max_length=50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_sanitize_string_sql_injection_patterns(self):
        """Test detection of SQL injection patterns"""
        dangerous_inputs = [
            "SELECT * FROM users",
            "1; DROP TABLE users",
            "admin' OR '1'='1",
            "'; DELETE FROM accounts--",
            "UNION SELECT password FROM users",
            "1 OR 1=1",
            "1 AND 1=1",
            "'; EXEC xp_cmdshell('dir')--",
        ]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(dangerous_input)
            assert "potentially dangerous patterns" in str(exc_info)

    def test_sanitize_string_xss_patterns(self):
        """Test detection of XSS patterns"""
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "<script type='text/javascript'>evil()</script>",
            "javascript:alert('XSS')",
            "<img onerror='alert(1)' src='x'>",
            "<body onload='alert(1)'>",
            "onclick='malicious()'",
        ]

        for xss_input in xss_inputs:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(xss_input)
            assert "potentially dangerous patterns" in str(exc_info)

    def test_sanitize_string_mixed_case_patterns(self):
        """Test case-insensitive pattern detection"""
        dangerous_inputs = ["SeLeCt * FrOm users", "JaVaScRiPt:alert(1)", "OnClIcK='evil()'"]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(dangerous_input)
            assert "potentially dangerous patterns" in str(exc_info)

    @patch("src.infrastructure.security.input_sanitizer.logger")
    def test_sanitize_string_logging(self, mock_logger):
        """Test that dangerous patterns are logged"""
        with pytest.raises(SanitizationError):
            InputSanitizer.sanitize_string("SELECT * FROM users")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "Dangerous pattern detected" in call_args


class TestSQLSanitization:
    """Test SQL-specific sanitization methods"""

    def test_sanitize_sql_identifier_valid(self):
        """Test valid SQL identifiers"""
        valid_identifiers = ["users", "user_accounts", "Account123", "a_very_long_table_name_123"]

        for identifier in valid_identifiers:
            result = InputSanitizer.sanitize_sql_identifier(identifier)
            assert result == identifier

    def test_sanitize_sql_identifier_invalid(self):
        """Test invalid SQL identifiers"""
        invalid_identifiers = [
            "123_starts_with_number",
            "has-dash",
            "has space",
            "has.dot",
            "has;semicolon",
            "'quoted'",
            "user$table",
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_sql_identifier(identifier)
            assert "Invalid SQL identifier" in str(exc_info)

    def test_sanitize_sql_identifier_reserved_words(self):
        """Test reserved SQL words"""
        reserved_words = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "select",
            "Select",
            "SeLeCt",  # Test case insensitivity
        ]

        for word in reserved_words:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_sql_identifier(word)
            assert "Reserved SQL word" in str(exc_info)

    def test_sanitize_sql_value_none(self):
        """Test sanitizing None value"""
        result = InputSanitizer.sanitize_sql_value(None)
        assert result == "NULL"

    def test_sanitize_sql_value_string(self):
        """Test sanitizing string values"""
        result = InputSanitizer.sanitize_sql_value("test")
        assert result == "'test'"

        # Test quote escaping
        result = InputSanitizer.sanitize_sql_value("O'Brien")
        assert result == "'O''Brien'"

        # Test backslash escaping
        result = InputSanitizer.sanitize_sql_value("path\\to\\file")
        assert result == "'path\\\\to\\\\file'"

    def test_sanitize_sql_value_numbers(self):
        """Test sanitizing numeric values"""
        result = InputSanitizer.sanitize_sql_value(123)
        assert result == "'123'"

        result = InputSanitizer.sanitize_sql_value(45.67)
        assert result == "'45.67'"

    def test_sanitize_sql_value_complex_escaping(self):
        """Test complex escaping scenarios"""
        # Both quotes and backslashes
        result = InputSanitizer.sanitize_sql_value("It's a 'test' with \\backslash\\")
        assert result == "'It''s a ''test'' with \\\\backslash\\\\'"


class TestFilenameSanitization:
    """Test filename sanitization"""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization"""
        result = InputSanitizer.sanitize_filename("document.pdf")
        assert result == "document.pdf"

        result = InputSanitizer.sanitize_filename("my-file_123.txt")
        assert result == "my-file_123.txt"

    def test_sanitize_filename_directory_traversal(self):
        """Test removal of directory traversal attempts"""
        result = InputSanitizer.sanitize_filename("../../../etc/passwd")
        assert result == "etc_passwd"

        result = InputSanitizer.sanitize_filename("..\\..\\windows\\system32\\config")
        assert result == "windows_system32_config"

        result = InputSanitizer.sanitize_filename("file/../other.txt")
        assert result == "file_other.txt"

    def test_sanitize_filename_special_cases(self):
        """Test special cases for .. removal"""
        # Just ".."
        result = InputSanitizer.sanitize_filename("..")
        assert result == ""

        # Starting with ../
        result = InputSanitizer.sanitize_filename("../file.txt")
        assert result == "file.txt"

        # Multiple .. at start
        result = InputSanitizer.sanitize_filename("../../file.txt")
        assert result == "file.txt"

        # .. in middle of filename (not a traversal)
        result = InputSanitizer.sanitize_filename("file..name.txt")
        assert result == "file..name.txt"

    def test_sanitize_filename_slashes(self):
        """Test slash replacement"""
        result = InputSanitizer.sanitize_filename("path/to/file.txt")
        assert result == "path_to_file.txt"

        result = InputSanitizer.sanitize_filename("C:\\Users\\file.txt")
        assert result == "C__Users_file.txt"

    def test_sanitize_filename_special_characters(self):
        """Test removal of special characters"""
        result = InputSanitizer.sanitize_filename("file<>:|?*.txt")
        assert result == "file______.txt"

        result = InputSanitizer.sanitize_filename("file@#$%.txt")
        assert result == "file____.txt"

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting"""
        # Without extension
        long_name = "a" * 300
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) == 255
        assert result == "a" * 255

        # With extension - should preserve extension
        long_name_with_ext = "a" * 300 + ".txt"
        result = InputSanitizer.sanitize_filename(long_name_with_ext)
        assert len(result) == 255
        assert result.endswith(".txt")
        assert result.startswith("a" * 251)  # 251 + 4 (.txt) = 255

        # With very long extension
        name_long_ext = "file." + "x" * 300
        result = InputSanitizer.sanitize_filename(name_long_ext)
        assert len(result) == 255

    def test_sanitize_filename_complex_extension_handling(self):
        """Test complex extension handling with length limits"""
        # Multiple dots in filename
        long_name = "a" * 250 + ".tar.gz"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) == 255
        assert result.endswith(".gz")  # Takes last extension

        # Extension too long for available space
        long_name = "a" * 254 + "." + "b" * 10
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) == 255


class TestTradingSymbolSanitization:
    """Test trading symbol sanitization"""

    def test_sanitize_symbol_valid(self):
        """Test valid trading symbols"""
        valid_symbols = ["AAPL", "MSFT", "BRK.B", "BRK-B", "VOO", "SPY500"]

        for symbol in valid_symbols:
            result = InputSanitizer.sanitize_symbol(symbol)
            assert result == symbol.upper()

    def test_sanitize_symbol_lowercase(self):
        """Test lowercase symbols are converted to uppercase"""
        result = InputSanitizer.sanitize_symbol("aapl")
        assert result == "AAPL"

        result = InputSanitizer.sanitize_symbol("Msft")
        assert result == "MSFT"

    def test_sanitize_symbol_invalid_characters(self):
        """Test symbols with invalid characters"""
        invalid_symbols = ["AAPL$", "MSFT@", "GO OG", "TEST_SYMBOL", "A*B", "C#D"]  # Space

        for symbol in invalid_symbols:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_symbol(symbol)
            assert "Invalid trading symbol" in str(exc_info)

    def test_sanitize_symbol_too_long(self):
        """Test symbols that are too long"""
        long_symbol = "A" * 21
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_symbol(long_symbol)
        assert "Symbol too long" in str(exc_info)

    def test_sanitize_symbol_sql_injection(self):
        """Test symbols containing SQL injection patterns"""
        dangerous_symbols = ["AAPL'; DROP TABLE--", "SELECT", "UNION"]

        for symbol in dangerous_symbols:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_symbol(symbol)
            assert "dangerous patterns" in str(exc_info) or "Invalid trading symbol" in str(
                exc_info
            )


class TestIdentifierSanitization:
    """Test general identifier sanitization"""

    def test_sanitize_identifier_valid(self):
        """Test valid identifiers"""
        valid_identifiers = ["user_id", "account-123", "TimeFrame1D", "api_key_v2"]

        for identifier in valid_identifiers:
            result = InputSanitizer.sanitize_identifier(identifier)
            assert result == identifier

    def test_sanitize_identifier_invalid_characters(self):
        """Test identifiers with invalid characters"""
        invalid_identifiers = ["user@id", "account#123", "time frame", "api.key", "value$"]  # Space

        for identifier in invalid_identifiers:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_identifier(identifier)
            assert "Invalid identifier" in str(exc_info)

    def test_sanitize_identifier_too_long(self):
        """Test identifiers that are too long"""
        long_identifier = "a" * 51
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_identifier(long_identifier)
        assert "Identifier too long" in str(exc_info)

    def test_sanitize_identifier_sql_injection(self):
        """Test identifiers with SQL injection patterns"""
        dangerous_identifiers = ["DROP_TABLE", "SELECT_FROM"]

        for identifier in dangerous_identifiers:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_identifier(identifier)
            assert "dangerous patterns" in str(exc_info)


class TestURLSanitization:
    """Test URL sanitization"""

    def test_sanitize_url_safe(self):
        """Test safe URLs pass through"""
        safe_urls = [
            "https://example.com",
            "http://localhost:8080/api",
            "https://api.example.com/v1/data",
            "ftp://files.example.com/download",
        ]

        for url in safe_urls:
            result = InputSanitizer.sanitize_url(url)
            # Should only do basic encoding
            assert "javascript:" not in result
            assert "data:" not in result

    def test_sanitize_url_dangerous_schemes(self):
        """Test dangerous URL schemes are rejected"""
        dangerous_urls = [
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "vbscript:msgbox('XSS')",
            "JavaScript:void(0)",  # Test case insensitivity
            "DATA:text/plain,test",
            "VBScript:test",
        ]

        for url in dangerous_urls:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_url(url)
            assert "Potentially dangerous URL scheme" in str(exc_info)

    def test_sanitize_url_encoding(self):
        """Test URL character encoding"""
        result = InputSanitizer.sanitize_url("https://example.com/path with spaces")
        assert result == "https://example.com/path%20with%20spaces"

        result = InputSanitizer.sanitize_url("https://example.com/<script>")
        assert result == "https://example.com/%3Cscript%3E"

        result = InputSanitizer.sanitize_url('https://example.com/"quoted"')
        assert result == "https://example.com/%22quoted%22"

    def test_sanitize_url_mixed_case_scheme(self):
        """Test case handling in URL schemes"""
        # Safe schemes should work regardless of case
        result = InputSanitizer.sanitize_url("HTTPS://EXAMPLE.COM")
        assert "javascript:" not in result

        result = InputSanitizer.sanitize_url("Http://Example.Com")
        assert "javascript:" not in result


class TestDangerousPatternDetection:
    """Test the internal dangerous pattern detection"""

    def test_contains_dangerous_pattern_sql(self):
        """Test SQL injection pattern detection"""
        dangerous_inputs = [
            "SELECT something",
            "INSERT INTO table",
            "UPDATE users SET",
            "DELETE FROM records",
            "DROP TABLE users",
            "CREATE TABLE new",
            "ALTER TABLE modify",
            "EXEC stored_proc",
            "EXECUTE command",
            "UNION SELECT",
            "FROM users WHERE",
            "-- comment",
            "/* comment */",
            "value; another",
            "1 OR 1=1",
            "1 AND 1=1",
            "xp_cmdshell",
            "sp_execute",
            "0x1234abcd",
        ]

        for dangerous_input in dangerous_inputs:
            assert InputSanitizer._contains_dangerous_pattern(dangerous_input) is True

    def test_contains_dangerous_pattern_xss(self):
        """Test XSS pattern detection"""
        xss_inputs = [
            "<script>code</script>",
            "<script type='text/javascript'>code</script>",
            "javascript:code",
            "onclick=function",
            "onmouseover=",
            "onerror=",
        ]

        for xss_input in xss_inputs:
            assert InputSanitizer._contains_dangerous_pattern(xss_input) is True

    def test_contains_dangerous_pattern_safe(self):
        """Test safe inputs don't trigger false positives"""
        safe_inputs = [
            "normal text",
            "user@example.com",
            "123-456-7890",
            "product_name_v2",
            "This is a sentence.",
            "json_data",
            "api_endpoint",
        ]

        for safe_input in safe_inputs:
            assert InputSanitizer._contains_dangerous_pattern(safe_input) is False

    def test_contains_dangerous_pattern_case_insensitive(self):
        """Test case-insensitive pattern matching"""
        assert InputSanitizer._contains_dangerous_pattern("select") is True
        assert InputSanitizer._contains_dangerous_pattern("SELECT") is True
        assert InputSanitizer._contains_dangerous_pattern("SeLeCt") is True
        assert InputSanitizer._contains_dangerous_pattern("JAVASCRIPT:") is True
        assert InputSanitizer._contains_dangerous_pattern("OnClick=") is True


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_strings(self):
        """Test handling of empty strings"""
        result = InputSanitizer.sanitize_string("")
        assert result == ""

        result = InputSanitizer.sanitize_sql_value("")
        assert result == "''"

        result = InputSanitizer.sanitize_filename("")
        assert result == ""

    def test_whitespace_only(self):
        """Test handling of whitespace-only strings"""
        result = InputSanitizer.sanitize_string("   ")
        assert result == ""

        result = InputSanitizer.sanitize_filename("   ")
        assert result == "___"

    def test_unicode_handling(self):
        """Test handling of Unicode characters"""
        result = InputSanitizer.sanitize_string("Hello 世界")
        assert result == "Hello 世界"

        result = InputSanitizer.sanitize_filename("文件.txt")
        assert result == "__.txt"  # Non-ASCII replaced with underscore

    def test_very_long_inputs(self):
        """Test handling of very long inputs"""
        long_string = "a" * 10000
        result = InputSanitizer.sanitize_string(long_string, max_length=1000)
        assert len(result) == 1000

    def test_special_sql_patterns(self):
        """Test special SQL patterns that should be caught"""
        # Hex values often used in SQL injection
        with pytest.raises(SanitizationError):
            InputSanitizer.sanitize_string("0x1234abcd")

        # SQL comments
        with pytest.raises(SanitizationError):
            InputSanitizer.sanitize_string("value--comment")

        with pytest.raises(SanitizationError):
            InputSanitizer.sanitize_string("value/*comment*/value")
