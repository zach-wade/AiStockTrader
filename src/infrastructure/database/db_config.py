"""
Database Configuration for AI Trader
Connects to the existing ai_trader PostgreSQL database
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ai_trader"  # IMPORTANT: Use ai_trader, not trading_db
    user: str = "zachwade"
    password: str = ""

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        # Check for DATABASE_URL first
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            parsed = urlparse(db_url)
            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else "ai_trader",
                user=parsed.username or "zachwade",
                password=parsed.password or "",
            )

        # Fall back to individual env vars
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ai_trader"),
            user=os.getenv("DB_USER", "zachwade"),
            password=os.getenv("DB_PASSWORD", ""),
        )

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        if self.password:
            return (
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            )
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        if self.password:
            return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"postgresql+asyncpg://{self.user}@{self.host}:{self.port}/{self.database}"


# Global instance
db_config = DatabaseConfig.from_env()


def get_db_config() -> DatabaseConfig:
    """Get database configuration."""
    return db_config


def test_connection():
    """Test database connection."""
    import psycopg2

    config = get_db_config()
    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
        )
        cur = conn.cursor()
        cur.execute("SELECT current_database(), COUNT(*) FROM companies")
        db_name, company_count = cur.fetchone()
        print(f"✅ Connected to database: {db_name}")
        print(f"✅ Found {company_count} companies in database")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
