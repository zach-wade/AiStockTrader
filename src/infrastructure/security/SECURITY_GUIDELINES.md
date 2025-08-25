# Security Guidelines for SQL Operations

## Overview

This document outlines critical security practices for SQL operations in the StockMonitoring application. **Manual SQL sanitization has been removed from this codebase for security reasons.**

## ⚠️ CRITICAL SECURITY RULE: NEVER MANUALLY SANITIZE SQL

**ALWAYS use parameterized queries for SQL operations. NEVER attempt to manually escape or sanitize SQL values.**

### Why Manual SQL Sanitization is Dangerous

1. **Incomplete Protection**: Manual escaping can miss edge cases and new attack vectors
2. **False Security**: Creates illusion of safety while remaining vulnerable
3. **Maintenance Risk**: Requires constant updates as new attacks emerge
4. **Human Error**: Easy to forget or misimplement in some code paths
5. **Encoding Issues**: Different character encodings can bypass manual escaping

## ✅ CORRECT: Parameterized Queries

### Python Examples

```python
# ✅ CORRECT - Using parameterized queries
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
cursor.execute("INSERT INTO orders (symbol, quantity) VALUES (%s, %s)", (symbol, quantity))
cursor.execute("UPDATE portfolio SET balance = %s WHERE user_id = %s", (new_balance, user_id))

# ✅ CORRECT - Multiple parameters
cursor.execute(
    "SELECT * FROM orders WHERE symbol = %s AND created_date > %s AND status = %s",
    (symbol, start_date, 'active')
)

# ✅ CORRECT - Using our QueryBuilder
from src.infrastructure.database.query_builder import QueryBuilder

query = (QueryBuilder()
    .select(['id', 'symbol', 'quantity'])
    .from_table('orders')
    .where('user_id = %s AND status = %s', [user_id, 'active'])
    .order_by('created_date', 'DESC')
    .build())

result = query.execute_with_cursor(cursor)
```

### ORM Examples

```python
# ✅ CORRECT - Using SQLAlchemy ORM
users = session.query(User).filter(User.id == user_id).all()
orders = session.query(Order).filter(
    Order.symbol == symbol,
    Order.quantity > min_quantity
).all()

# ✅ CORRECT - SQLAlchemy Core with parameterized queries
result = connection.execute(
    text("SELECT * FROM orders WHERE symbol = :symbol"),
    {'symbol': symbol}
)
```

## ❌ INCORRECT: Manual SQL Escaping (NEVER DO THIS)

```python
# ❌ NEVER DO THIS - Manual string concatenation
query = f"SELECT * FROM users WHERE name = '{user_name}'"  # VULNERABLE!

# ❌ NEVER DO THIS - Manual escaping
escaped_name = user_name.replace("'", "''")
query = f"SELECT * FROM users WHERE name = '{escaped_name}'"  # STILL VULNERABLE!

# ❌ NEVER DO THIS - String formatting
query = "SELECT * FROM users WHERE id = {}".format(user_id)  # VULNERABLE!

# ❌ NEVER DO THIS - % formatting
query = "SELECT * FROM users WHERE name = '%s'" % (user_name,)  # VULNERABLE!
```

## SQL Identifiers vs Values

### Identifiers (Table/Column Names)

Use the `InputSanitizer.sanitize_sql_identifier()` method for table and column names:

```python
# ✅ CORRECT - Validating SQL identifiers
from src.infrastructure.security.input_sanitizer import InputSanitizer

table_name = InputSanitizer.sanitize_sql_identifier(user_input_table)
column_name = InputSanitizer.sanitize_sql_identifier(user_input_column)

# Then use in parameterized query (identifiers cannot be parameterized)
query = f"SELECT {column_name} FROM {table_name} WHERE id = %s"
cursor.execute(query, (record_id,))
```

**Note**: SQL identifiers (table names, column names) cannot be parameterized, so they require validation. However, this should be rare - most applications use fixed table/column names.

### Values (Data)

**ALWAYS** use parameterized queries for values:

```python
# ✅ CORRECT - Parameterized values
cursor.execute("SELECT * FROM orders WHERE symbol = %s", (symbol_value,))
```

## Input Validation vs SQL Security

### Use InputValidator for Type Safety

```python
from src.infrastructure.security.input_validation import InputValidator

validator = InputValidator()

# Validate data types and formats
symbol = validator.validate_trading_symbol(user_input, 'symbol')
quantity = validator.validate_decimal(user_input, 'quantity', min_value=Decimal('0.001'))
email = validator.validate_email(user_input, 'email')

# Then use validated data in parameterized queries
cursor.execute(
    "INSERT INTO orders (symbol, quantity, user_email) VALUES (%s, %s, %s)",
    (symbol, quantity, email)
)
```

## Database Connection Security

### Connection String Security

```python
# ✅ CORRECT - Use environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
connection = psycopg2.connect(DATABASE_URL)

# ❌ NEVER DO THIS - Hardcoded credentials
connection = psycopg2.connect(
    "postgresql://user:password@localhost/db"  # NEVER HARDCODE!
)
```

### Connection Pool Settings

```python
# ✅ CORRECT - Secure connection pool
pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=20,
    dsn=DATABASE_URL,
    connect_timeout=10,
    application_name="StockMonitoring"
)
```

## Error Handling Security

### Don't Leak Information in Errors

```python
# ✅ CORRECT - Generic error messages to users
try:
    cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
    result = cursor.fetchone()
except psycopg2.Error as e:
    logger.error(f"Database error: {e}")  # Log detailed error
    raise ValueError("Invalid order ID")  # Generic error to user

# ❌ NEVER DO THIS - Exposing database details
try:
    cursor.execute(query, params)
except psycopg2.Error as e:
    raise Exception(f"Database error: {e}")  # EXPOSES INTERNAL DETAILS!
```

## Migration Security

### Secure Database Migrations

```python
# ✅ CORRECT - Parameterized migration queries
def upgrade():
    op.execute(
        "INSERT INTO config (key, value) VALUES (%s, %s)",
        ('default_currency', 'USD')
    )

# ❌ NEVER DO THIS - String concatenation in migrations
def upgrade():
    value = get_config_value()
    op.execute(f"INSERT INTO config (key, value) VALUES ('setting', '{value}')")
```

## Testing Security

### Security Tests

```python
def test_sql_injection_prevention():
    """Test that parameterized queries prevent SQL injection."""
    malicious_input = "'; DROP TABLE users; --"

    # This should safely treat the malicious input as data
    cursor.execute(
        "SELECT * FROM users WHERE name = %s",
        (malicious_input,)
    )

    # Verify the table still exists
    cursor.execute("SELECT COUNT(*) FROM users")
    assert cursor.fetchone()[0] >= 0  # Should not fail

def test_input_validation():
    """Test input validation catches malicious patterns."""
    from src.infrastructure.security.input_validation import InputValidator

    validator = InputValidator()

    with pytest.raises(ValidationError):
        validator.validate_email("<script>alert('xss')</script>")
```

## Monitoring and Logging

### Log Security Events

```python
import logging

security_logger = logging.getLogger('security')

def log_suspicious_activity(user_id, activity, details):
    """Log security-related events."""
    security_logger.warning(
        f"Suspicious activity from user {user_id}: {activity}",
        extra={
            'user_id': user_id,
            'activity': activity,
            'details': details,
            'timestamp': datetime.utcnow()
        }
    )
```

## Code Review Checklist

When reviewing code, check for:

- [ ] All SQL queries use parameterized queries (`%s` placeholders)
- [ ] No string concatenation or formatting in SQL queries
- [ ] SQL identifiers are validated using `sanitize_sql_identifier()`
- [ ] Input validation is performed before database operations
- [ ] Database errors don't leak sensitive information
- [ ] Connection strings use environment variables
- [ ] Security events are logged appropriately

## Tools and Libraries

### Recommended Libraries

- **psycopg2**: PostgreSQL adapter with parameterized query support
- **SQLAlchemy**: ORM with built-in SQL injection protection
- **Our QueryBuilder**: Type-safe parameterized query builder

### Avoid These Anti-Patterns

- Manual string escaping functions
- Dynamic SQL generation with string concatenation
- Using `%` formatting or `.format()` with SQL
- Trusted input assumptions

## Emergency Response

If you discover SQL injection vulnerabilities:

1. **Immediately** stop using the vulnerable code
2. **Assess** what data may have been compromised
3. **Fix** the vulnerability using parameterized queries
4. **Review** similar code patterns for the same issue
5. **Test** the fix thoroughly
6. **Deploy** the fix as quickly as possible
7. **Monitor** for any unusual database activity

## Additional Resources

- [OWASP SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [PostgreSQL Security Documentation](https://www.postgresql.org/docs/current/security.html)
- [Python DB-API Parameterized Queries](https://peps.python.org/pep-0249/)

## Questions?

If you have questions about secure SQL practices, contact the security team or review this documentation. **When in doubt, always use parameterized queries.**

---

**Remember: The only safe way to include user data in SQL queries is through parameterized queries. No exceptions.**
