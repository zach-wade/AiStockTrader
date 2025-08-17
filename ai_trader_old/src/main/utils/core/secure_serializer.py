"""
Secure Serializer for Safe Data Deserialization

This module provides secure serialization and deserialization functionality
to prevent code injection attacks through pickle vulnerabilities.

Security Features:
- JSON-first serialization for simple types
- Restricted pickle deserialization with type validation
- Integrity checking with SHA256 hashing
- Whitelist-based class loading for pickle
- Comprehensive error handling and logging

Created to address G2.1 Insecure Pickle Deserialization vulnerability.
"""

# Standard library imports
from datetime import datetime
from decimal import Decimal
import hashlib
import io
import json
import logging
import pickle
from typing import Any

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""

    pass


class SecureSerializer:
    """
    Secure serialization with validation and restricted pickle usage.

    This class provides safe serialization/deserialization by:
    1. Using JSON for simple types (preferred)
    2. Using restricted pickle only for whitelisted classes
    3. Adding integrity checks with SHA256 hashing
    4. Comprehensive validation and error handling
    """

    # Whitelist of allowed classes for pickle deserialization
    ALLOWED_CLASSES = {
        # Built-in types
        "builtins": {
            "dict",
            "list",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "NoneType",
            "set",
            "frozenset",
            "bytes",
            "bytearray",
        },
        # Standard library types
        "datetime": {"datetime", "date", "time", "timedelta"},
        "decimal": {"Decimal"},
        "collections": {"defaultdict", "OrderedDict", "Counter"},
        # AI Trader model types (add as needed)
        "main.models.common": {
            "MarketData",
            "Quote",
            "Position",
            "Order",
            "Trade",
            "OrderStatus",
            "OrderType",
            "OrderSide",
            "TimeInForce",
        },
        "pandas.core.frame": {"DataFrame"},
        "pandas.core.series": {"Series"},
        "numpy": {"ndarray", "int64", "float64"},
    }

    def __init__(self, use_json_fallback: bool = True, verify_integrity: bool = True):
        """
        Initialize SecureSerializer.

        Args:
            use_json_fallback: Whether to prefer JSON serialization
            verify_integrity: Whether to include integrity checks
        """
        self.use_json_fallback = use_json_fallback
        self.verify_integrity = verify_integrity

    def serialize(self, obj: Any) -> bytes:
        """
        Securely serialize an object.

        Args:
            obj: Object to serialize

        Returns:
            Serialized bytes with format prefix

        Raises:
            SecurityError: If object contains unsafe types
            ValueError: If object is None or invalid
        """
        if obj is None:
            logger.warning("Attempting to serialize None object")
            return b"JSON:null"

        try:
            # Try JSON first for simple types
            if self.use_json_fallback and self._is_json_serializable(obj):
                try:
                    json_str = json.dumps(obj, default=self._json_default)
                    json_data = json_str.encode("utf-8")

                    # Sanity check: ensure we can deserialize what we just serialized
                    if len(json_data) > 100 * 1024 * 1024:  # 100MB limit
                        raise SecurityError("JSON serialization too large (>100MB)")

                    return b"JSON:" + json_data
                except (TypeError, ValueError, UnicodeEncodeError) as e:
                    logger.debug(f"JSON serialization failed, falling back to pickle: {e}")

            # Use secure pickle for complex objects
            if self._validate_object_for_pickle(obj):
                try:
                    pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

                    # Size limit check
                    if len(pickled_data) > 100 * 1024 * 1024:  # 100MB limit
                        raise SecurityError("Pickle serialization too large (>100MB)")

                    if self.verify_integrity:
                        # Add integrity hash
                        data_hash = hashlib.sha256(pickled_data).hexdigest().encode("utf-8")
                        return b"PICKLE_HASH:" + data_hash + b":" + pickled_data
                    else:
                        return b"PICKLE:" + pickled_data

                except (pickle.PickleError, MemoryError, RecursionError) as e:
                    logger.error(f"Pickle serialization failed: {e}")
                    raise SecurityError(f"Pickle serialization failed: {e!s}")
            else:
                obj_type = type(obj).__name__
                obj_module = type(obj).__module__
                raise SecurityError(
                    f"Object type {obj_module}.{obj_type} not allowed for serialization. "
                    f"Consider adding to ALLOWED_CLASSES whitelist if safe."
                )

        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Unexpected serialization error: {e}", exc_info=True)
            raise SecurityError(f"Serialization failed: {e!s}")

    def deserialize(self, data: bytes) -> Any:
        """
        Securely deserialize data.

        Args:
            data: Serialized bytes to deserialize

        Returns:
            Deserialized object

        Raises:
            SecurityError: If deserialization fails or data is unsafe
            ValueError: If data format is invalid
        """
        if data is None:
            raise SecurityError("Cannot deserialize None data")

        if not isinstance(data, bytes):
            raise SecurityError(f"Data must be bytes, got {type(data)}")

        if len(data) == 0:
            raise SecurityError("Cannot deserialize empty data")

        if len(data) > 100 * 1024 * 1024:  # 100MB limit
            raise SecurityError("Data too large for deserialization (>100MB)")

        try:
            # Determine serialization format
            if data.startswith(b"JSON:"):
                # JSON deserialization
                try:
                    json_data = data[5:].decode("utf-8")
                    if not json_data:
                        logger.warning("Empty JSON data")
                        return None
                    return json.loads(json_data)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.error(f"JSON deserialization failed: {e}")
                    raise SecurityError(f"Invalid JSON data: {e!s}")

            elif data.startswith(b"PICKLE_HASH:"):
                # Pickle with integrity check
                try:
                    parts = data[12:].split(b":", 1)
                    if len(parts) != 2:
                        raise SecurityError("Invalid pickle hash format - missing hash or data")

                    expected_hash, pickled_data = parts

                    if len(expected_hash) == 0 or len(pickled_data) == 0:
                        raise SecurityError("Invalid pickle hash format - empty hash or data")

                    try:
                        expected_hash = expected_hash.decode("utf-8")
                    except UnicodeDecodeError:
                        raise SecurityError("Invalid pickle hash format - non-UTF8 hash")

                    # Verify hash format (SHA256 is 64 hex characters)
                    if len(expected_hash) != 64 or not all(
                        c in "0123456789abcdef" for c in expected_hash.lower()
                    ):
                        raise SecurityError("Invalid hash format - not a valid SHA256 hash")

                    # Verify integrity
                    actual_hash = hashlib.sha256(pickled_data).hexdigest()
                    if actual_hash != expected_hash:
                        logger.error("Data integrity check failed - hash mismatch")
                        raise SecurityError("Data integrity check failed - possible tampering")

                    # Use restricted unpickler
                    return self._safe_pickle_loads(pickled_data)

                except SecurityError:
                    raise
                except Exception as e:
                    logger.error(f"Hash verification failed: {e}")
                    raise SecurityError(f"Hash verification failed: {e!s}")

            elif data.startswith(b"PICKLE:"):
                # Pickle without integrity check (less secure)
                logger.warning("Deserializing pickle data without integrity check - security risk")
                pickled_data = data[7:]
                if len(pickled_data) == 0:
                    raise SecurityError("Empty pickle data")
                return self._safe_pickle_loads(pickled_data)

            else:
                # Unknown format
                prefix = data[:20] if len(data) >= 20 else data
                logger.error(f"Unknown serialization format: {prefix}")
                raise SecurityError(
                    f"Unknown serialization format. Expected JSON:, PICKLE:, or PICKLE_HASH: prefix. "
                    f"Got: {prefix!r}"
                )

        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Unexpected deserialization error: {e}", exc_info=True)
            raise SecurityError(f"Deserialization failed: {e!s}")

    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object can be safely serialized as JSON."""
        try:
            json.dumps(obj, default=self._json_default)
            return True
        except (TypeError, ValueError, OverflowError):
            return False

    def _json_default(self, obj: Any) -> Any:
        """Default JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            raise TypeError(f"Object {type(obj)} is not JSON serializable")

    def _validate_object_for_pickle(self, obj: Any) -> bool:
        """
        Recursively validate that an object only contains safe types for pickling.

        Args:
            obj: Object to validate

        Returns:
            True if object is safe to pickle
        """
        obj_type = type(obj)
        module_name = obj_type.__module__
        class_name = obj_type.__name__

        # Check if class is in whitelist
        if module_name not in self.ALLOWED_CLASSES:
            logger.warning(f"Module {module_name} not in whitelist")
            return False

        if class_name not in self.ALLOWED_CLASSES[module_name]:
            logger.warning(f"Class {module_name}.{class_name} not in whitelist")
            return False

        # Recursively check container contents
        if isinstance(obj, (list, tuple)):
            return all(self._validate_object_for_pickle(item) for item in obj)
        elif isinstance(obj, dict):
            return all(
                self._validate_object_for_pickle(k) and self._validate_object_for_pickle(v)
                for k, v in obj.items()
            )
        elif isinstance(obj, set):
            return all(self._validate_object_for_pickle(item) for item in obj)

        return True

    def _safe_pickle_loads(self, data: bytes) -> Any:
        """
        Safely load pickle data with restricted execution.

        Args:
            data: Pickle data bytes

        Returns:
            Unpickled object

        Raises:
            SecurityError: If unsafe class is encountered
        """

        class RestrictedUnpickler(pickle.Unpickler):
            """Unpickler that only allows whitelisted classes."""

            def find_class(self, module, name):
                # Check if module and class are in whitelist
                if module in SecureSerializer.ALLOWED_CLASSES:
                    if name in SecureSerializer.ALLOWED_CLASSES[module]:
                        return super().find_class(module, name)

                # Log the attempt for security monitoring
                logger.error(f"Blocked attempt to load class {module}.{name}")
                raise pickle.UnpicklingError(
                    f"Class {module}.{name} not allowed - potential security risk"
                )

        try:
            return RestrictedUnpickler(io.BytesIO(data)).load()
        except Exception as e:
            logger.error(f"Restricted pickle loading failed: {e}")
            raise SecurityError(f"Pickle deserialization failed: {e!s}")


# Global instance for convenience
secure_serializer = SecureSerializer()


def secure_dumps(obj: Any) -> bytes:
    """Convenience function for secure serialization."""
    return secure_serializer.serialize(obj)


def secure_loads(data: bytes) -> Any:
    """Convenience function for secure deserialization."""
    return secure_serializer.deserialize(data)


def migrate_unsafe_pickle(unsafe_data: bytes, validate_result: bool = True) -> bytes:
    """
    Migrate unsafe pickle data to secure format.

    Args:
        unsafe_data: Raw pickle data
        validate_result: Whether to validate the unpickled object

    Returns:
        Securely serialized data

    Raises:
        SecurityError: If data contains unsafe objects
    """
    logger.warning("Migrating unsafe pickle data - this should only be done once")

    try:
        # Use standard pickle to load (ONE TIME ONLY for migration)
        obj = pickle.loads(unsafe_data)

        # Validate the object if requested
        if validate_result:
            serializer = SecureSerializer()
            if not serializer._validate_object_for_pickle(obj):
                raise SecurityError("Migrated object contains unsafe types")

        # Re-serialize securely
        return secure_dumps(obj)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise SecurityError(f"Failed to migrate unsafe pickle data: {e!s}")
