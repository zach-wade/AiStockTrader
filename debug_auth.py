#!/usr/bin/env python3
"""Debug authentication issues."""

import logging
import asyncio
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Import auth components
from src.infrastructure.auth.models import Base
from src.infrastructure.auth.jwt_service import JWTService
from src.infrastructure.auth.rbac_service import RBACService
from src.infrastructure.auth.services.user_service import UserService
from tests.integration.auth.test_auth_flow import test_app


def setup_test_environment():
    """Setup test environment."""
    # Create test database
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=True  # Enable SQL logging
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    
    # Create mock Redis client
    redis_mock = Mock()
    storage = {}
    
    def mock_setex(key, ttl, value):
        storage[key] = value
        return True
    
    def mock_get(key):
        return storage.get(key)
    
    def mock_delete(key):
        if key in storage:
            del storage[key]
        return True
    
    redis_mock.setex = Mock(side_effect=mock_setex)
    redis_mock.get = Mock(side_effect=mock_get)
    redis_mock.delete = Mock(side_effect=mock_delete)
    redis_mock.sadd = Mock()
    redis_mock.expire = Mock()
    redis_mock.zremrangebyscore = Mock()
    redis_mock.zcard = Mock(return_value=0)
    redis_mock.zadd = Mock()
    
    # Create services
    jwt_service = JWTService(redis_client=redis_mock)
    rbac_service = RBACService(db_session=db)
    user_service = UserService(db_session=db, jwt_service=jwt_service, require_email_verification=False)
    
    # Initialize default roles
    asyncio.run(rbac_service.initialize_default_roles_and_permissions())
    db.commit()
    
    # Create test app
    app = test_app(db, jwt_service, rbac_service, user_service, redis_mock)
    client = TestClient(app)
    
    return client, db, user_service


def main():
    """Main debug function."""
    client, db, user_service = setup_test_environment()
    
    # Test 1: Registration
    print("=== Testing Registration ===")
    reg_response = client.post(
        "/auth/register",
        json={
            "email": "testuser@example.com",
            "username": "testuser",
            "password": "TestP@ssw0rd123!",
        },
    )
    print(f"Registration Status: {reg_response.status_code}")
    print(f"Registration Response: {reg_response.json()}")
    
    if reg_response.status_code != 201:
        print("Registration failed, stopping test")
        return
    
    # Check user in database
    print("\n=== Checking User in Database ===")
    from src.infrastructure.auth.models import User
    user = db.query(User).filter_by(username="testuser").first()
    if user:
        print(f"User found: {user.username}, email: {user.email}")
        print(f"Email verified: {user.email_verified}")
        print(f"Password hash: {user.password_hash[:50]}...")
        print(f"User ID: {user.id}")
        
        # Test password verification directly
        print("\n=== Testing Password Verification ===")
        from src.infrastructure.auth.services.password_service import PasswordService
        password_service = PasswordService()
        is_valid = password_service.verify_password("TestP@ssw0rd123!", user.password_hash)
        print(f"Direct password verification: {is_valid}")
        
    else:
        print("User not found in database!")
        return
    
    # Test 2: Login
    print("\n=== Testing Login ===")
    login_response = client.post(
        "/auth/login",
        data={
            "username": "testuser",
            "password": "TestP@ssw0rd123!",
        },
    )
    print(f"Login Status: {login_response.status_code}")
    print(f"Login Response: {login_response.json()}")
    
    # Test 3: Login with email
    print("\n=== Testing Login with Email ===")
    login_response2 = client.post(
        "/auth/login",
        data={
            "username": "testuser@example.com",
            "password": "TestP@ssw0rd123!",
        },
    )
    print(f"Login with Email Status: {login_response2.status_code}")
    print(f"Login with Email Response: {login_response2.json()}")


if __name__ == "__main__":
    main()