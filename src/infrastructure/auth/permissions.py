"""
Permission definitions and matrix for the AI Trading System.

This module defines all system permissions and provides
a centralized permission matrix for role-based access control.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class Resource(str, Enum):
    """System resources that can be protected."""

    # Trading resources
    PORTFOLIO = "portfolio"
    TRADES = "trades"
    ORDERS = "orders"
    STRATEGIES = "strategies"
    POSITIONS = "positions"

    # User management
    USERS = "users"
    ROLES = "roles"
    API_KEYS = "api_keys"
    SESSIONS = "sessions"

    # System resources
    SYSTEM = "system"
    AUDIT = "audit"
    REPORTS = "reports"
    MARKET_DATA = "market_data"
    SETTINGS = "settings"
    NOTIFICATIONS = "notifications"

    # Risk management
    RISK = "risk"
    LIMITS = "limits"
    COMPLIANCE = "compliance"


class Action(str, Enum):
    """Actions that can be performed on resources."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Trading actions
    EXECUTE = "execute"
    CANCEL = "cancel"
    MODIFY = "modify"
    APPROVE = "approve"
    REJECT = "reject"

    # Administrative actions
    ADMIN = "admin"
    CONFIGURE = "configure"
    EXPORT = "export"
    IMPORT = "import"
    AUDIT = "audit"

    # Special actions
    OVERRIDE = "override"
    EMERGENCY_STOP = "emergency_stop"
    RESET = "reset"
    GENERATE = "generate"
    REVOKE = "revoke"


@dataclass(frozen=True)
class Permission:
    """
    Represents a permission as a combination of resource and action.
    """

    resource: Resource
    action: Action
    description: str = ""

    def __str__(self) -> str:
        """String representation of permission."""
        return f"{self.resource.value}:{self.action.value}"

    def to_string(self) -> str:
        """Convert to string format for storage."""
        return str(self)

    @classmethod
    def from_string(cls, permission_string: str) -> "Permission":
        """Create Permission from string format."""
        parts = permission_string.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid permission format: {permission_string}")

        resource_str, action_str = parts
        try:
            resource = Resource(resource_str)
            action = Action(action_str)
        except ValueError as e:
            raise ValueError(f"Invalid permission components: {e}")

        return cls(resource=resource, action=action)


class PermissionMatrix:
    """
    Centralized permission matrix defining all system permissions
    and role assignments.
    """

    # Define all system permissions
    PERMISSIONS: ClassVar[dict[str, Permission]] = {
        # Portfolio permissions
        "portfolio:read": Permission(Resource.PORTFOLIO, Action.READ, "View portfolio data"),
        "portfolio:update": Permission(
            Resource.PORTFOLIO, Action.UPDATE, "Update portfolio settings"
        ),
        "portfolio:delete": Permission(Resource.PORTFOLIO, Action.DELETE, "Delete portfolio"),
        "portfolio:export": Permission(Resource.PORTFOLIO, Action.EXPORT, "Export portfolio data"),
        # Trading permissions
        "trades:read": Permission(Resource.TRADES, Action.READ, "View trade history"),
        "trades:execute": Permission(Resource.TRADES, Action.EXECUTE, "Execute new trades"),
        "trades:cancel": Permission(Resource.TRADES, Action.CANCEL, "Cancel pending trades"),
        "trades:modify": Permission(Resource.TRADES, Action.MODIFY, "Modify existing trades"),
        "trades:approve": Permission(
            Resource.TRADES, Action.APPROVE, "Approve trades (for limits)"
        ),
        # Order permissions
        "orders:read": Permission(Resource.ORDERS, Action.READ, "View orders"),
        "orders:create": Permission(Resource.ORDERS, Action.CREATE, "Create new orders"),
        "orders:cancel": Permission(Resource.ORDERS, Action.CANCEL, "Cancel orders"),
        "orders:modify": Permission(Resource.ORDERS, Action.MODIFY, "Modify orders"),
        # Position permissions
        "positions:read": Permission(Resource.POSITIONS, Action.READ, "View positions"),
        "positions:update": Permission(Resource.POSITIONS, Action.UPDATE, "Update positions"),
        "positions:delete": Permission(Resource.POSITIONS, Action.DELETE, "Close positions"),
        # Strategy permissions
        "strategies:read": Permission(Resource.STRATEGIES, Action.READ, "View trading strategies"),
        "strategies:create": Permission(
            Resource.STRATEGIES, Action.CREATE, "Create new strategies"
        ),
        "strategies:update": Permission(Resource.STRATEGIES, Action.UPDATE, "Modify strategies"),
        "strategies:delete": Permission(Resource.STRATEGIES, Action.DELETE, "Delete strategies"),
        "strategies:execute": Permission(Resource.STRATEGIES, Action.EXECUTE, "Execute strategies"),
        # User management permissions
        "users:read": Permission(Resource.USERS, Action.READ, "View user profiles"),
        "users:create": Permission(Resource.USERS, Action.CREATE, "Create new users"),
        "users:update": Permission(Resource.USERS, Action.UPDATE, "Modify user data"),
        "users:delete": Permission(Resource.USERS, Action.DELETE, "Delete user accounts"),
        # Role permissions
        "roles:read": Permission(Resource.ROLES, Action.READ, "View roles"),
        "roles:create": Permission(Resource.ROLES, Action.CREATE, "Create new roles"),
        "roles:update": Permission(Resource.ROLES, Action.UPDATE, "Modify roles"),
        "roles:delete": Permission(Resource.ROLES, Action.DELETE, "Delete roles"),
        # API key permissions
        "api_keys:create": Permission(Resource.API_KEYS, Action.CREATE, "Create API keys"),
        "api_keys:read": Permission(Resource.API_KEYS, Action.READ, "View API keys"),
        "api_keys:revoke": Permission(Resource.API_KEYS, Action.REVOKE, "Revoke API keys"),
        # System permissions
        "system:admin": Permission(Resource.SYSTEM, Action.ADMIN, "Full system administration"),
        "system:configure": Permission(
            Resource.SYSTEM, Action.CONFIGURE, "Configure system settings"
        ),
        "system:emergency_stop": Permission(
            Resource.SYSTEM, Action.EMERGENCY_STOP, "Emergency trading halt"
        ),
        "system:reset": Permission(Resource.SYSTEM, Action.RESET, "Reset system components"),
        # Audit permissions
        "audit:read": Permission(Resource.AUDIT, Action.READ, "View audit logs"),
        "audit:export": Permission(Resource.AUDIT, Action.EXPORT, "Export audit logs"),
        "audit:audit": Permission(Resource.AUDIT, Action.AUDIT, "Perform audits"),
        # Report permissions
        "reports:read": Permission(Resource.REPORTS, Action.READ, "View reports"),
        "reports:generate": Permission(Resource.REPORTS, Action.GENERATE, "Generate reports"),
        "reports:export": Permission(Resource.REPORTS, Action.EXPORT, "Export reports"),
        # Market data permissions
        "market_data:read": Permission(Resource.MARKET_DATA, Action.READ, "View market data"),
        "market_data:import": Permission(Resource.MARKET_DATA, Action.IMPORT, "Import market data"),
        # Risk permissions
        "risk:read": Permission(Resource.RISK, Action.READ, "View risk metrics"),
        "risk:override": Permission(Resource.RISK, Action.OVERRIDE, "Override risk limits"),
        "risk:configure": Permission(Resource.RISK, Action.CONFIGURE, "Configure risk parameters"),
        # Limits permissions
        "limits:read": Permission(Resource.LIMITS, Action.READ, "View trading limits"),
        "limits:update": Permission(Resource.LIMITS, Action.UPDATE, "Update trading limits"),
        "limits:override": Permission(Resource.LIMITS, Action.OVERRIDE, "Override trading limits"),
        # Compliance permissions
        "compliance:read": Permission(Resource.COMPLIANCE, Action.READ, "View compliance data"),
        "compliance:audit": Permission(
            Resource.COMPLIANCE, Action.AUDIT, "Perform compliance audits"
        ),
        "compliance:export": Permission(
            Resource.COMPLIANCE, Action.EXPORT, "Export compliance reports"
        ),
        # Settings permissions
        "settings:read": Permission(Resource.SETTINGS, Action.READ, "View system settings"),
        "settings:update": Permission(Resource.SETTINGS, Action.UPDATE, "Update system settings"),
        # Notification permissions
        "notifications:read": Permission(Resource.NOTIFICATIONS, Action.READ, "View notifications"),
        "notifications:create": Permission(
            Resource.NOTIFICATIONS, Action.CREATE, "Create notifications"
        ),
        "notifications:delete": Permission(
            Resource.NOTIFICATIONS, Action.DELETE, "Delete notifications"
        ),
    }

    # Role permission mappings
    ROLE_PERMISSIONS: ClassVar[dict[str, list[str]]] = {
        "ADMIN": [
            "system:admin",  # Full admin has all permissions implicitly
        ],
        "TRADER": [
            # Portfolio access
            "portfolio:read",
            "portfolio:update",
            "portfolio:export",
            # Trading operations
            "trades:read",
            "trades:execute",
            "trades:cancel",
            "trades:modify",
            # Order management
            "orders:read",
            "orders:create",
            "orders:cancel",
            "orders:modify",
            # Position management
            "positions:read",
            "positions:update",
            # Strategy execution
            "strategies:read",
            "strategies:execute",
            # Reports and market data
            "reports:read",
            "reports:generate",
            "market_data:read",
            # Risk monitoring
            "risk:read",
            "limits:read",
            # Settings (read-only)
            "settings:read",
            # Notifications
            "notifications:read",
            "notifications:create",
            # API keys (own only)
            "api_keys:create",
            "api_keys:read",
            "api_keys:revoke",
        ],
        "VIEWER": [
            # Read-only access to trading data
            "portfolio:read",
            "trades:read",
            "orders:read",
            "positions:read",
            "strategies:read",
            "reports:read",
            "market_data:read",
            "risk:read",
            "limits:read",
            "settings:read",
            "notifications:read",
        ],
        "API_USER": [
            # Programmatic trading access
            "portfolio:read",
            "trades:read",
            "trades:execute",
            "orders:read",
            "orders:create",
            "orders:cancel",
            "positions:read",
            "strategies:execute",
            "market_data:read",
            "risk:read",
            "limits:read",
        ],
        "RISK_MANAGER": [
            # Risk management operations
            "portfolio:read",
            "trades:read",
            "trades:approve",
            "orders:read",
            "positions:read",
            "risk:read",
            "risk:configure",
            "risk:override",
            "limits:read",
            "limits:update",
            "limits:override",
            "reports:read",
            "reports:generate",
            "compliance:read",
            "compliance:audit",
            "compliance:export",
            "audit:read",
            "settings:read",
        ],
        "COMPLIANCE_OFFICER": [
            # Compliance and audit operations
            "portfolio:read",
            "trades:read",
            "orders:read",
            "positions:read",
            "users:read",
            "audit:read",
            "audit:export",
            "audit:audit",
            "compliance:read",
            "compliance:audit",
            "compliance:export",
            "reports:read",
            "reports:generate",
            "reports:export",
            "settings:read",
        ],
        "STRATEGY_DEVELOPER": [
            # Strategy development and testing
            "portfolio:read",
            "trades:read",
            "orders:read",
            "positions:read",
            "strategies:read",
            "strategies:create",
            "strategies:update",
            "strategies:delete",
            "market_data:read",
            "reports:read",
            "risk:read",
            "settings:read",
        ],
    }

    # Critical operations requiring additional authentication
    CRITICAL_OPERATIONS: ClassVar[list[str]] = [
        "system:admin",
        "system:emergency_stop",
        "system:reset",
        "users:delete",
        "roles:delete",
        "risk:override",
        "limits:override",
        "compliance:audit",
    ]

    # Operations that should be logged for audit
    AUDITABLE_OPERATIONS: ClassVar[list[str]] = [
        "trades:execute",
        "trades:cancel",
        "trades:approve",
        "orders:create",
        "orders:cancel",
        "positions:update",
        "positions:delete",
        "strategies:execute",
        "users:create",
        "users:update",
        "users:delete",
        "roles:create",
        "roles:update",
        "roles:delete",
        "api_keys:create",
        "api_keys:revoke",
        "risk:override",
        "limits:override",
        "system:configure",
        "system:emergency_stop",
        "system:reset",
    ]

    @classmethod
    def get_role_permissions(cls, role: str) -> list[str]:
        """Get all permissions for a role."""
        role_upper = role.upper()
        if role_upper == "ADMIN":
            # Admin has all permissions
            return list(cls.PERMISSIONS.keys())
        return cls.ROLE_PERMISSIONS.get(role_upper, [])

    @classmethod
    def is_critical_operation(cls, permission: str) -> bool:
        """Check if permission is a critical operation."""
        return permission in cls.CRITICAL_OPERATIONS

    @classmethod
    def is_auditable_operation(cls, permission: str) -> bool:
        """Check if permission should be audited."""
        return permission in cls.AUDITABLE_OPERATIONS

    @classmethod
    def validate_permission(cls, permission_string: str) -> bool:
        """Validate if a permission string is valid."""
        return permission_string in cls.PERMISSIONS

    @classmethod
    def get_permission_description(cls, permission_string: str) -> str:
        """Get description for a permission."""
        permission = cls.PERMISSIONS.get(permission_string)
        return permission.description if permission else ""
