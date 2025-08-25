"""
Comprehensive unit tests for input sanitization module.

Tests all sanitization methods including SQL injection prevention,
XSS prevention, filename sanitization, and path traversal prevention.
"""

import html

import pytest

from src.infrastructure.security.input_sanitizer import InputSanitizer, SanitizationError


class TestSanitizationError:
    """Test SanitizationError exception."""

    def test_sanitization_error(self):
        """Test SanitizationError creation."""
        error = SanitizationError("Input contains malicious content")
        assert str(error) == "Input contains malicious content"
        assert isinstance(error, Exception)


class TestInputSanitizerStringMethods:
    """Test InputSanitizer string sanitization methods."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        result = InputSanitizer.sanitize_string("  Hello World  ")
        assert result == html.escape("Hello World")

    def test_sanitize_string_with_html(self):
        """Test sanitizing HTML content."""
        input_str = "<b>Bold</b> text"
        result = InputSanitizer.sanitize_string(input_str)
        assert result == "&lt;b&gt;Bold&lt;/b&gt; text"

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

    def test_sanitize_string_sql_injection_patterns(self):
        """Test detection of SQL injection patterns."""
        sql_injections = [
            "'; DROP TABLE users; --",
            "SELECT * FROM users",
            "admin'--",
            "' UNION SELECT * FROM passwords",
            "DELETE FROM users WHERE 1=1",
            "' OR 1=1--",
            "' AND 1=1--",
            "'; EXEC xp_cmdshell('dir')",
            "'; EXECUTE sp_configure",
        ]

        for injection in sql_injections:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(injection)
            assert "dangerous patterns" in str(exc_info).lower()

    def test_sanitize_string_xss_patterns(self):
        """Test detection of XSS patterns."""
        xss_attempts = [
            "<script>alert('XSS')</script>",
            "<script src='evil.js'></script>",
            "javascript:alert('XSS')",
            "<img onerror='alert(1)' src='x'>",
            "<body onload='alert(1)'>",
            "<svg onload='alert(1)'>",
        ]

        for xss in xss_attempts:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(xss)
            assert "dangerous patterns" in str(exc_info).lower()

    def test_sanitize_string_mixed_case_patterns(self):
        """Test detection with mixed case patterns."""
        dangerous_inputs = [
            "SeLeCt * FrOm users",
            "DeLeTe FROM accounts",
            "JavaScRiPt:alert(1)",
            "<ScRiPt>alert(1)</ScRiPt>",
        ]

        for input_str in dangerous_inputs:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_string(input_str)
            assert "dangerous patterns" in str(exc_info).lower()

    def test_sanitize_string_safe_inputs(self):
        """Test that safe inputs pass through."""
        safe_inputs = [
            "John Doe",
            "user@example.com",
            "Product description with normal text",
            "123 Main Street",
            "Price: $99.99",
        ]

        for input_str in safe_inputs:
            result = InputSanitizer.sanitize_string(input_str)
            assert result == html.escape(input_str)


class TestSqlIdentifierSanitization:
    """Test SQL identifier sanitization."""

    def test_sanitize_sql_identifier_valid(self):
        """Test valid SQL identifiers."""
        valid_identifiers = ["users", "user_accounts", "OrderItems", "table123", "my_table_name"]

        for identifier in valid_identifiers:
            result = InputSanitizer.sanitize_sql_identifier(identifier)
            assert result == identifier

    def test_sanitize_sql_identifier_invalid_characters(self):
        """Test identifiers with invalid characters."""
        invalid_identifiers = [
            "table-name",  # Hyphen not allowed
            "table.name",  # Dot not allowed
            "table name",  # Space not allowed
            "123table",  # Can't start with number
            "table$name",  # Special character
            "table;drop",  # Semicolon
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_sql_identifier(identifier)
            assert "Invalid SQL identifier" in str(exc_info)

    def test_sanitize_sql_identifier_reserved_words(self):
        """Test SQL reserved words."""
        reserved_words = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "select",  # Case insensitive
            "Select",
        ]

        for word in reserved_words:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_sql_identifier(word)
            assert "Reserved SQL word" in str(exc_info)

    def test_sanitize_sql_identifier_empty(self):
        """Test empty identifier."""
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_sql_identifier("")
        assert "Invalid SQL identifier" in str(exc_info)


class TestSqlValueSanitization:
    """Test SQL value sanitization."""

    def test_sanitize_sql_value_none(self):
        """Test sanitizing None value."""
        result = InputSanitizer.sanitize_sql_value(None)
        assert result == "NULL"

    def test_sanitize_sql_value_string(self):
        """Test sanitizing string values."""
        result = InputSanitizer.sanitize_sql_value("John's data")
        assert result == "'John''s data'"

        result = InputSanitizer.sanitize_sql_value("Regular text")
        assert result == "'Regular text'"

    def test_sanitize_sql_value_quotes_escaping(self):
        """Test proper quote escaping."""
        result = InputSanitizer.sanitize_sql_value("O'Brien")
        assert result == "'O''Brien'"

        result = InputSanitizer.sanitize_sql_value("It's a 'quoted' string")
        assert result == "'It''s a ''quoted'' string'"

    def test_sanitize_sql_value_backslash_escaping(self):
        """Test backslash escaping."""
        result = InputSanitizer.sanitize_sql_value("path\\to\\file")
        assert result == "'path\\\\to\\\\file'"

        result = InputSanitizer.sanitize_sql_value("Line1\\nLine2")
        assert result == "'Line1\\\\nLine2'"

    def test_sanitize_sql_value_numbers(self):
        """Test sanitizing numeric values."""
        result = InputSanitizer.sanitize_sql_value(123)
        assert result == "'123'"

        result = InputSanitizer.sanitize_sql_value(45.67)
        assert result == "'45.67'"

    def test_sanitize_sql_value_boolean(self):
        """Test sanitizing boolean values."""
        result = InputSanitizer.sanitize_sql_value(True)
        assert result == "'True'"

        result = InputSanitizer.sanitize_sql_value(False)
        assert result == "'False'"


class TestFilenameSanitization:
    """Test filename sanitization."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = InputSanitizer.sanitize_filename("document.pdf")
        assert result == "document.pdf"

        result = InputSanitizer.sanitize_filename("my-file_123.txt")
        assert result == "my-file_123.txt"

    def test_sanitize_filename_path_traversal(self):
        """Test path traversal prevention."""
        dangerous_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "file/../../../etc/shadow",
            "../../secret.txt",
            "../",
            "..",
            "normal..txt",  # This should keep the dots in the middle
        ]

        results = [
            "etc_passwd",
            "windows_system32_config",
            "file_etc_shadow",
            "secret.txt",
            "_",
            "",
            "normal..txt",
        ]

        for filename, expected in zip(dangerous_filenames, results):
            result = InputSanitizer.sanitize_filename(filename)
            assert result == expected

    def test_sanitize_filename_directory_separators(self):
        """Test removal of directory separators."""
        result = InputSanitizer.sanitize_filename("path/to/file.txt")
        assert result == "path_to_file.txt"

        result = InputSanitizer.sanitize_filename("path\\to\\file.txt")
        assert result == "path_to_file.txt"

    def test_sanitize_filename_special_characters(self):
        """Test removal of special characters."""
        result = InputSanitizer.sanitize_filename("file<>:|?*.txt")
        assert result == "file______.txt"

        result = InputSanitizer.sanitize_filename("file@#$%^&*().pdf")
        assert result == "file_________.pdf"

    def test_sanitize_filename_unicode(self):
        """Test handling of unicode characters."""
        result = InputSanitizer.sanitize_filename("文件名.txt")
        assert result == "_____.txt"  # Non-ASCII replaced with underscores

        result = InputSanitizer.sanitize_filename("café.pdf")
        assert result == "caf_.pdf"

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting."""
        long_name = "a" * 300 + ".txt"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) == 255
        assert result.endswith(".txt")
        assert result.startswith("a" * 251)

    def test_sanitize_filename_long_extension(self):
        """Test handling of long extensions."""
        filename = "file." + "x" * 300
        result = InputSanitizer.sanitize_filename(filename)
        assert len(result) == 255

    def test_sanitize_filename_dots_at_edges(self):
        """Test handling dots at start/end."""
        result = InputSanitizer.sanitize_filename("..file.txt")
        assert result == "file.txt"

        result = InputSanitizer.sanitize_filename("../file.txt")
        assert result == "file.txt"


class TestSymbolSanitization:
    """Test trading symbol sanitization."""

    def test_sanitize_symbol_valid(self):
        """Test valid trading symbols."""
        valid_symbols = ["AAPL", "MSFT", "BRK.B", "BRK-B", "SPY", "VOO", "QQQ123"]

        for symbol in valid_symbols:
            result = InputSanitizer.sanitize_symbol(symbol)
            assert result == symbol.upper()

    def test_sanitize_symbol_lowercase(self):
        """Test lowercase symbols are converted to uppercase."""
        result = InputSanitizer.sanitize_symbol("aapl")
        assert result == "AAPL"

        result = InputSanitizer.sanitize_symbol("MsFt")
        assert result == "MSFT"

    def test_sanitize_symbol_invalid_characters(self):
        """Test symbols with invalid characters."""
        invalid_symbols = ["AAPL$", "MSFT@", "GOOG!", "FB#", "AMZN*", "TSLA()"]

        for symbol in invalid_symbols:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_symbol(symbol)
            assert "Invalid trading symbol" in str(exc_info)

    def test_sanitize_symbol_too_long(self):
        """Test symbols that are too long."""
        long_symbol = "A" * 21
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_symbol(long_symbol)
        assert "Symbol too long" in str(exc_info)

    def test_sanitize_symbol_sql_injection(self):
        """Test symbols containing SQL injection attempts."""
        malicious_symbols = ["AAPL'; DROP TABLE--", "MSFT OR 1=1", "SELECT * FROM"]

        for symbol in malicious_symbols:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_symbol(symbol)
            assert "dangerous patterns" in str(exc_info).lower() or "Invalid trading symbol" in str(
                exc_info
            )


class TestIdentifierSanitization:
    """Test general identifier sanitization."""

    def test_sanitize_identifier_valid(self):
        """Test valid identifiers."""
        valid_identifiers = ["user_id", "api-key", "session123", "time_frame", "ORDER_TYPE"]

        for identifier in valid_identifiers:
            result = InputSanitizer.sanitize_identifier(identifier)
            assert result == identifier

    def test_sanitize_identifier_invalid_characters(self):
        """Test identifiers with invalid characters."""
        invalid_identifiers = [
            "user@id",
            "api.key",
            "session!",
            "time frame",  # Space
            "order$type",
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_identifier(identifier)
            assert "Invalid identifier" in str(exc_info)

    def test_sanitize_identifier_too_long(self):
        """Test identifier length limit."""
        long_identifier = "a" * 51
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_identifier(long_identifier)
        assert "Identifier too long" in str(exc_info)

    def test_sanitize_identifier_sql_injection(self):
        """Test identifiers with SQL injection patterns."""
        malicious = ["user_id; DROP TABLE users", "session_OR_1=1"]

        for identifier in malicious:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_identifier(identifier)
            assert "dangerous patterns" in str(exc_info).lower() or "Invalid identifier" in str(
                exc_info
            )


class TestUrlSanitization:
    """Test URL sanitization."""

    def test_sanitize_url_basic(self):
        """Test basic URL sanitization."""
        result = InputSanitizer.sanitize_url("https://example.com")
        assert result == "https://example.com"

        result = InputSanitizer.sanitize_url("http://api.example.com/endpoint")
        assert result == "http://api.example.com/endpoint"

    def test_sanitize_url_spaces(self):
        """Test URL with spaces."""
        result = InputSanitizer.sanitize_url("https://example.com/path with spaces")
        assert result == "https://example.com/path%20with%20spaces"

    def test_sanitize_url_special_characters(self):
        """Test URL with special characters."""
        result = InputSanitizer.sanitize_url("https://example.com/<script>")
        assert result == "https://example.com/%3Cscript%3E"

        result = InputSanitizer.sanitize_url('https://example.com/"test"')
        assert result == "https://example.com/%22test%22"

    def test_sanitize_url_javascript_scheme(self):
        """Test javascript: URL scheme."""
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_url("javascript:alert('XSS')")
        assert "dangerous URL scheme" in str(exc_info)

    def test_sanitize_url_data_scheme(self):
        """Test data: URL scheme."""
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_url("data:text/html,<script>alert('XSS')</script>")
        assert "dangerous URL scheme" in str(exc_info)

    def test_sanitize_url_vbscript_scheme(self):
        """Test vbscript: URL scheme."""
        with pytest.raises(SanitizationError) as exc_info:
            InputSanitizer.sanitize_url("vbscript:msgbox('XSS')")
        assert "dangerous URL scheme" in str(exc_info)

    def test_sanitize_url_case_insensitive_scheme(self):
        """Test case insensitive scheme detection."""
        dangerous_urls = [
            "JavaScript:alert(1)",
            "JAVASCRIPT:alert(1)",
            "Data:text/html,test",
            "VBScript:test",
        ]

        for url in dangerous_urls:
            with pytest.raises(SanitizationError) as exc_info:
                InputSanitizer.sanitize_url(url)
            assert "dangerous URL scheme" in str(exc_info)


class TestDangerousPatternDetection:
    """Test _contains_dangerous_pattern method."""

    def test_sql_keywords_detection(self):
        """Test detection of SQL keywords."""
        sql_keywords = [
            "SELECT something",
            "INSERT INTO table",
            "UPDATE table SET",
            "DELETE FROM table",
            "DROP TABLE users",
            "CREATE TABLE new",
            "ALTER TABLE modify",
            "EXEC procedure",
            "EXECUTE command",
            "UNION SELECT",
        ]

        for text in sql_keywords:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_sql_comment_detection(self):
        """Test detection of SQL comments."""
        sql_comments = ["value--comment", "value/*comment*/", "value;command", "value|command"]

        for text in sql_comments:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_sql_logic_detection(self):
        """Test detection of SQL logic patterns."""
        sql_logic = ["1 OR 1=1", "1 AND 1=1", "value OR 2=2", "test AND 5=5"]

        for text in sql_logic:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_xss_script_detection(self):
        """Test detection of script tags."""
        scripts = [
            "<script>code</script>",
            "<script src='evil.js'></script>",
            "<SCRIPT>alert(1)</SCRIPT>",
        ]

        for text in scripts:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_xss_javascript_detection(self):
        """Test detection of javascript: protocol."""
        javascript_urls = ["javascript:alert(1)", "JavaScript:code", "JAVASCRIPT:test"]

        for text in javascript_urls:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_xss_event_handler_detection(self):
        """Test detection of event handlers."""
        event_handlers = ["onclick='alert(1)'", "onload=code", "onerror='test'", "onmouseover=func"]

        for text in event_handlers:
            assert InputSanitizer._contains_dangerous_pattern(text) is True

    def test_safe_patterns(self):
        """Test that safe patterns are not detected."""
        safe_texts = [
            "Normal text without issues",
            "User selected an option",
            "Insert coin to continue",
            "Update your profile",
            "Delete this item?",
            "Create a new account",
            "Alternate route available",
            "Union Station",
            "This is from the database",
            "Click anywhere to continue",
        ]

        for text in safe_texts:
            assert InputSanitizer._contains_dangerous_pattern(text) is False


class TestEdgeCasesAndSpecialScenarios:
    """Test edge cases and special scenarios."""

    def test_empty_string_sanitization(self):
        """Test sanitizing empty strings."""
        result = InputSanitizer.sanitize_string("")
        assert result == ""

        result = InputSanitizer.sanitize_string("   ")
        assert result == ""

    def test_whitespace_only_sanitization(self):
        """Test sanitizing whitespace-only strings."""
        result = InputSanitizer.sanitize_string("\t\n\r ")
        assert result == ""

    def test_very_long_string_truncation(self):
        """Test truncation of very long strings."""
        long_string = "a" * 10000
        result = InputSanitizer.sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_nested_dangerous_patterns(self):
        """Test nested dangerous patterns."""
        nested = "<script><script>alert(1)</script></script>"
        with pytest.raises(SanitizationError):
            InputSanitizer.sanitize_string(nested)

    def test_obfuscated_patterns(self):
        """Test obfuscated dangerous patterns."""
        obfuscated = [
            "SEL" + "ECT * FR" + "OM users",  # String concatenation attempt
            "java\nscript:alert(1)",  # Newline insertion
            "java\tscript:alert(1)",  # Tab insertion
        ]

        # These should still be caught if properly reassembled
        for text in obfuscated:
            # The sanitizer should catch the reassembled pattern
            if "SELECT" in text.upper() or "javascript" in text.lower().replace("\n", "").replace(
                "\t", ""
            ):
                with pytest.raises(SanitizationError):
                    InputSanitizer.sanitize_string(text)

    def test_unicode_normalization(self):
        """Test handling of unicode characters."""
        # Unicode characters that might be used to bypass filters
        unicode_tests = [
            "ｓｅｌｅｃｔ",  # Full-width characters
            "ＳＥＬＥＣＴ",
        ]

        for text in unicode_tests:
            # These are safe as they're not ASCII SELECT
            result = InputSanitizer.sanitize_string(text)
            assert result == html.escape(text)

    def test_null_byte_injection(self):
        """Test null byte injection attempts."""
        null_byte_tests = [
            "file.txt\x00.jpg",
            "data\x00<script>",
        ]

        for text in null_byte_tests:
            # Should handle null bytes safely
            result = InputSanitizer.sanitize_string(text)
            # The exact behavior depends on how null bytes are handled
            assert "\x00" in result or result == html.escape(text.replace("\x00", ""))


class TestComplexAttackVectors:
    """Test complex and combined attack vectors."""

    def test_polyglot_attacks(self):
        """Test polyglot attack strings."""
        polyglots = [
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
            '"><script>alert(1)</script>',
            "' OR '1'='1' /*",
            "admin' AND '1'='1' --",
        ]

        for attack in polyglots:
            with pytest.raises(SanitizationError):
                InputSanitizer.sanitize_string(attack)

    def test_combined_sql_xss(self):
        """Test combined SQL and XSS attacks."""
        combined = [
            "'; DROP TABLE users; <script>alert(1)</script>--",
            "1 UNION SELECT '<script>alert(1)</script>'",
        ]

        for attack in combined:
            with pytest.raises(SanitizationError):
                InputSanitizer.sanitize_string(attack)

    def test_encoding_bypasses(self):
        """Test various encoding bypass attempts."""
        encoded_attacks = [
            "%3Cscript%3Ealert(1)%3C/script%3E",  # URL encoded
            "&#60;script&#62;alert(1)&#60;/script&#62;",  # HTML entity encoded
        ]

        for attack in encoded_attacks:
            # These might pass through as they're encoded
            # The sanitizer focuses on the raw patterns
            result = InputSanitizer.sanitize_string(attack)
            assert result == html.escape(attack)
