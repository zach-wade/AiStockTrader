"""
Audit event formatters for standardized output formatting.

This module provides various formatters for audit events to support different
output formats, compliance requirements, and integration needs.
"""

import csv
import io
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from .events import ComplianceRegulation
from .exceptions import AuditFormattingError


class AuditFormatter(ABC):
    """
    Abstract base class for audit event formatters.

    Defines the interface that all formatters must implement,
    ensuring consistent formatting behavior across different output formats.
    """

    def __init__(self, include_sensitive_data: bool = True, timezone_format: str = "iso") -> None:
        """
        Initialize audit formatter.

        Args:
            include_sensitive_data: Whether to include sensitive data in output
            timezone_format: Format for timezone representation ('iso', 'utc', 'local')
        """
        self.include_sensitive_data = include_sensitive_data
        self.timezone_format = timezone_format

    @abstractmethod
    def format(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """
        Format audit event data.

        Args:
            event_data: Raw audit event data

        Returns:
            Formatted event data
        """
        pass

    @abstractmethod
    def format_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format batch of audit events.

        Args:
            events: List of raw audit event data

        Returns:
            List of formatted event data
        """
        pass

    def _sanitize_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove or mask sensitive data if configured to do so."""
        if self.include_sensitive_data:
            return data

        # Create a copy to avoid modifying original data
        sanitized = data.copy()

        # Define sensitive fields that should be masked
        sensitive_fields = [
            "user_id",
            "session_id",
            "ip_address",
            "user_agent",
            "api_key",
            "password",
            "token",
            "secret",
        ]

        for field in sensitive_fields:
            if field in sanitized:
                if isinstance(sanitized[field], str) and len(sanitized[field]) > 4:
                    # Mask all but first 4 characters
                    sanitized[field] = sanitized[field][:4] + "*" * (len(sanitized[field]) - 4)
                else:
                    sanitized[field] = "***"

        return sanitized

    def _format_datetime(self, dt_str: str) -> str:
        """Format datetime string according to timezone format setting."""
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

            if self.timezone_format == "iso":
                return dt.isoformat()
            elif self.timezone_format == "utc":
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            elif self.timezone_format == "local":
                local_dt = dt.astimezone()
                return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            else:
                return dt_str  # Return original if format unknown

        except (ValueError, AttributeError):
            return dt_str  # Return original if parsing fails

    def _serialize_decimal(self, value: Any) -> Any:
        """Convert Decimal objects to strings for JSON serialization."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, dict):
            return {k: self._serialize_decimal(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_decimal(item) for item in value]
        else:
            return value


class JSONFormatter(AuditFormatter):
    """
    JSON formatter for audit events.

    Provides clean, structured JSON output suitable for log analysis
    tools, SIEM systems, and API responses.
    """

    def __init__(
        self,
        include_sensitive_data: bool = True,
        timezone_format: str = "iso",
        pretty_print: bool = False,
        sort_keys: bool = True,
    ):
        """
        Initialize JSON formatter.

        Args:
            include_sensitive_data: Whether to include sensitive data
            timezone_format: Format for timezone representation
            pretty_print: Whether to format JSON with indentation
            sort_keys: Whether to sort keys in JSON output
        """
        super().__init__(include_sensitive_data, timezone_format)
        self.pretty_print = pretty_print
        self.sort_keys = sort_keys

    def format(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Format single audit event as JSON-ready dictionary."""
        try:
            # Sanitize sensitive data
            sanitized_data = self._sanitize_sensitive_data(event_data)

            # Format timestamp
            if "timestamp" in sanitized_data:
                sanitized_data["timestamp"] = self._format_datetime(sanitized_data["timestamp"])

            # Handle Decimal serialization
            formatted_data = self._serialize_decimal(sanitized_data)

            # Add formatting metadata
            formatted_data["_formatter"] = {
                "type": "json",
                "version": "1.0",
                "formatted_at": datetime.utcnow().isoformat() + "Z",
            }

            return cast(dict[str, Any], formatted_data)

        except Exception as e:
            raise AuditFormattingError(
                f"Failed to format event as JSON: {e}", formatter_type="json", event_data=event_data
            )

    def format_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format batch of audit events as JSON-ready list."""
        formatted_events = []

        for i, event_data in enumerate(events):
            try:
                formatted_event = self.format(event_data)
                formatted_events.append(formatted_event)
            except Exception as e:
                # Log error but continue with other events
                formatted_events.append(
                    {
                        "error": f"Formatting failed: {e}",
                        "original_event_id": event_data.get("event_id", f"batch_item_{i}"),
                        "_formatter": {
                            "type": "json",
                            "version": "1.0",
                            "error": True,
                            "formatted_at": datetime.utcnow().isoformat() + "Z",
                        },
                    }
                )

        return formatted_events

    def to_json_string(self, event_data: dict[str, Any]) -> str:
        """Convert formatted event to JSON string."""
        formatted_data = self.format(event_data)

        if self.pretty_print:
            return json.dumps(
                formatted_data, indent=2, sort_keys=self.sort_keys, ensure_ascii=False
            )
        else:
            separators = (",", ":") if not self.pretty_print else (",", ": ")
            return json.dumps(
                formatted_data, sort_keys=self.sort_keys, separators=separators, ensure_ascii=False
            )


class ComplianceFormatter(AuditFormatter):
    """
    Compliance-focused formatter for regulatory reporting.

    Formats audit events according to specific regulatory requirements
    and includes all necessary compliance fields.
    """

    def __init__(
        self,
        regulation: ComplianceRegulation,
        jurisdiction: str = "US",
        include_sensitive_data: bool = True,
        timezone_format: str = "iso",
    ):
        """
        Initialize compliance formatter.

        Args:
            regulation: Target compliance regulation
            jurisdiction: Regulatory jurisdiction
            include_sensitive_data: Whether to include sensitive data
            timezone_format: Format for timezone representation
        """
        super().__init__(include_sensitive_data, timezone_format)
        self.regulation = regulation
        self.jurisdiction = jurisdiction

    def format(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Format audit event for compliance reporting."""
        try:
            # Start with sanitized base data
            formatted_data = self._sanitize_sensitive_data(event_data)

            # Format timestamp
            if "timestamp" in formatted_data:
                formatted_data["timestamp"] = self._format_datetime(formatted_data["timestamp"])

            # Add compliance-specific headers
            compliance_header = {
                "regulation": self.regulation.value,
                "jurisdiction": self.jurisdiction,
                "report_generated_at": datetime.utcnow().isoformat() + "Z",
                "compliance_version": "1.0",
                "regulatory_requirements": self._get_regulatory_requirements(),
            }

            # Restructure data according to regulation requirements
            if self.regulation == ComplianceRegulation.SOX:
                formatted_data = self._format_sox_compliance(formatted_data)
            elif self.regulation == ComplianceRegulation.GDPR:
                formatted_data = self._format_gdpr_compliance(formatted_data)
            elif self.regulation == ComplianceRegulation.MIFID_II:
                formatted_data = self._format_mifid_compliance(formatted_data)
            elif self.regulation == ComplianceRegulation.SEC:
                formatted_data = self._format_sec_compliance(formatted_data)
            elif self.regulation == ComplianceRegulation.CFTC:
                formatted_data = self._format_cftc_compliance(formatted_data)

            # Add compliance header
            formatted_data["compliance"] = compliance_header

            return formatted_data

        except Exception as e:
            raise AuditFormattingError(
                f"Failed to format event for compliance: {e}",
                formatter_type=f"compliance_{self.regulation.value}",
                event_data=event_data,
            )

    def format_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format batch of events for compliance reporting."""
        formatted_events = []
        batch_header = {
            "batch_id": f"compliance_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "batch_size": len(events),
            "regulation": self.regulation.value,
            "jurisdiction": self.jurisdiction,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        for event_data in events:
            formatted_event = self.format(event_data)
            formatted_event["batch_info"] = batch_header
            formatted_events.append(formatted_event)

        return formatted_events

    def _get_regulatory_requirements(self) -> list[str]:
        """Get regulatory requirements for the specified regulation."""
        requirements_map = {
            ComplianceRegulation.SOX: [
                "financial_data_integrity",
                "audit_trail_completeness",
                "access_controls",
                "change_management",
            ],
            ComplianceRegulation.GDPR: [
                "data_minimization",
                "consent_tracking",
                "right_to_erasure",
                "data_portability",
            ],
            ComplianceRegulation.MIFID_II: [
                "transaction_reporting",
                "best_execution",
                "client_classification",
                "record_keeping",
            ],
            ComplianceRegulation.SEC: [
                "trade_reporting",
                "market_surveillance",
                "investor_protection",
                "record_retention",
            ],
            ComplianceRegulation.CFTC: [
                "swap_data_reporting",
                "position_limits",
                "risk_management",
                "record_keeping",
            ],
        }

        return requirements_map.get(self.regulation, [])

    def _format_sox_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format data according to SOX requirements."""
        sox_data = data.copy()

        # Add SOX-specific fields
        sox_data["sox_compliance"] = {
            "internal_control_assertion": True,
            "financial_data_impact": self._assess_financial_impact(data),
            "management_assertion": "accurate_and_complete",
            "audit_trail_id": data.get("event_id"),
            "control_effectiveness": "operating_effectively",
        }

        return sox_data

    def _format_gdpr_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format data according to GDPR requirements."""
        gdpr_data = data.copy()

        # Identify personal data
        personal_data_fields = self._identify_personal_data(data)

        gdpr_data["gdpr_compliance"] = {
            "lawful_basis": "legitimate_interest",
            "personal_data_fields": personal_data_fields,
            "retention_period_days": 365,
            "data_subject_rights": ["access", "rectification", "erasure"],
            "processing_purpose": "audit_logging",
            "data_minimization_applied": len(personal_data_fields) <= 3,
        }

        return gdpr_data

    def _format_mifid_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format data according to MiFID II requirements."""
        mifid_data = data.copy()

        # Check if this is a transaction that needs MiFID reporting
        is_reportable = self._is_mifid_reportable_transaction(data)

        mifid_data["mifid_compliance"] = {
            "reportable_transaction": is_reportable,
            "venue_identification": "INTERNAL",
            "transaction_identification": data.get("event_id"),
            "execution_timestamp": data.get("timestamp"),
            "best_execution_applied": True,
            "client_classification": "professional",
        }

        return mifid_data

    def _format_sec_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format data according to SEC requirements."""
        sec_data = data.copy()

        sec_data["sec_compliance"] = {
            "rule_compliance": ["17a-4", "17a-3"],  # Record keeping rules
            "record_type": self._classify_sec_record_type(data),
            "retention_requirement_years": 5,
            "examination_readiness": True,
            "regulatory_reporting_required": self._requires_sec_reporting(data),
        }

        return sec_data

    def _format_cftc_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format data according to CFTC requirements."""
        cftc_data = data.copy()

        cftc_data["cftc_compliance"] = {
            "swap_data_repository_reporting": False,  # Most audit events aren't swap related
            "position_reporting_required": False,
            "record_type": "audit_trail",
            "retention_period_years": 5,
            "regulatory_examination_ready": True,
        }

        return cftc_data

    def _assess_financial_impact(self, data: dict[str, Any]) -> str:
        """Assess financial impact for SOX compliance."""
        resource_type = data.get("resource_type", "")

        if resource_type in ["order", "position", "portfolio"]:
            return "direct_financial_impact"
        elif resource_type in ["risk", "configuration"]:
            return "indirect_financial_impact"
        else:
            return "no_financial_impact"

    def _identify_personal_data(self, data: dict[str, Any]) -> list[str]:
        """Identify personal data fields for GDPR compliance."""
        personal_data_fields = []

        if data.get("user_id"):
            personal_data_fields.append("user_id")
        if data.get("context", {}).get("ip_address"):
            personal_data_fields.append("ip_address")
        if data.get("context", {}).get("user_agent"):
            personal_data_fields.append("user_agent")

        return personal_data_fields

    def _is_mifid_reportable_transaction(self, data: dict[str, Any]) -> bool:
        """Determine if transaction is reportable under MiFID II."""
        event_type = data.get("event_type", "")
        resource_type = data.get("resource_type", "")

        return resource_type == "order" and event_type in ["create", "fill", "modify"]

    def _classify_sec_record_type(self, data: dict[str, Any]) -> str:
        """Classify record type for SEC compliance."""
        resource_type = data.get("resource_type", "")

        if resource_type == "order":
            return "order_record"
        elif resource_type == "position":
            return "position_record"
        elif resource_type == "portfolio":
            return "account_record"
        else:
            return "administrative_record"

    def _requires_sec_reporting(self, data: dict[str, Any]) -> bool:
        """Determine if record requires SEC reporting."""
        resource_type = data.get("resource_type", "")
        is_critical = data.get("is_critical", False)

        return resource_type in ["order", "position"] or is_critical


class XMLFormatter(AuditFormatter):
    """
    XML formatter for audit events.

    Provides XML output suitable for legacy systems and specific
    compliance requirements that mandate XML format.
    """

    def __init__(
        self,
        include_sensitive_data: bool = True,
        timezone_format: str = "iso",
        pretty_print: bool = True,
        root_element: str = "audit_event",
    ):
        """
        Initialize XML formatter.

        Args:
            include_sensitive_data: Whether to include sensitive data
            timezone_format: Format for timezone representation
            pretty_print: Whether to format XML with indentation
            root_element: Name of the root XML element
        """
        super().__init__(include_sensitive_data, timezone_format)
        self.pretty_print = pretty_print
        self.root_element = root_element

    def format(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Format audit event as XML structure."""
        try:
            # Sanitize sensitive data
            sanitized_data = self._sanitize_sensitive_data(event_data)

            # Create XML structure
            xml_string = self._dict_to_xml(sanitized_data)

            return {
                "format": "xml",
                "content": xml_string,
                "encoding": "utf-8",
                "_formatter": {
                    "type": "xml",
                    "version": "1.0",
                    "formatted_at": datetime.utcnow().isoformat() + "Z",
                },
            }

        except Exception as e:
            raise AuditFormattingError(
                f"Failed to format event as XML: {e}", formatter_type="xml", event_data=event_data
            )

    def format_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format batch of audit events as XML."""
        formatted_events = []

        for event_data in events:
            formatted_event = self.format(event_data)
            formatted_events.append(formatted_event)

        return formatted_events

    def _dict_to_xml(self, data: dict[str, Any]) -> str:
        """Convert dictionary to XML string."""
        root = ET.Element(self.root_element)
        self._add_dict_to_element(root, data)

        if self.pretty_print:
            self._indent_xml(root)

        return ET.tostring(root, encoding="unicode")

    def _add_dict_to_element(self, parent: ET.Element, data: dict[str, Any]) -> None:
        """Recursively add dictionary data to XML element."""
        for key, value in data.items():
            # Sanitize key name for XML
            clean_key = self._sanitize_xml_key(key)

            if isinstance(value, dict):
                child = ET.SubElement(parent, clean_key)
                self._add_dict_to_element(child, value)
            elif isinstance(value, list):
                for item in value:
                    child = ET.SubElement(parent, clean_key)
                    if isinstance(item, dict):
                        self._add_dict_to_element(child, item)
                    else:
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, clean_key)
                child.text = str(value) if value is not None else ""

    def _sanitize_xml_key(self, key: str) -> str:
        """Sanitize dictionary key for use as XML element name."""
        # Replace invalid XML name characters
        sanitized = key.replace(" ", "_").replace("-", "_")

        # Ensure it starts with a letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "_" + sanitized

        return sanitized or "unnamed"

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """Add indentation to XML for pretty printing."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent_xml(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class CSVFormatter(AuditFormatter):
    """
    CSV formatter for audit events.

    Provides tabular output suitable for spreadsheet analysis
    and reporting tools.
    """

    def __init__(
        self,
        include_sensitive_data: bool = True,
        timezone_format: str = "iso",
        field_order: list[str] | None = None,
    ):
        """
        Initialize CSV formatter.

        Args:
            include_sensitive_data: Whether to include sensitive data
            timezone_format: Format for timezone representation
            field_order: Preferred order of fields in CSV output
        """
        super().__init__(include_sensitive_data, timezone_format)
        self.field_order = field_order or [
            "event_id",
            "timestamp",
            "event_type",
            "resource_type",
            "resource_id",
            "action",
            "user_id",
            "severity",
            "is_critical",
        ]

    def format(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Format audit event for CSV output."""
        try:
            # Sanitize sensitive data
            sanitized_data = self._sanitize_sensitive_data(event_data)

            # Flatten nested structures
            flattened_data = self._flatten_dict(sanitized_data)

            # Format timestamp
            if "timestamp" in flattened_data:
                flattened_data["timestamp"] = self._format_datetime(flattened_data["timestamp"])

            return {
                "format": "csv",
                "data": flattened_data,
                "_formatter": {
                    "type": "csv",
                    "version": "1.0",
                    "formatted_at": datetime.utcnow().isoformat() + "Z",
                },
            }

        except Exception as e:
            raise AuditFormattingError(
                f"Failed to format event as CSV: {e}", formatter_type="csv", event_data=event_data
            )

    def format_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format batch of events as CSV."""
        formatted_events = []

        for event_data in events:
            formatted_event = self.format(event_data)
            formatted_events.append(formatted_event)

        return formatted_events

    def to_csv_string(self, events: list[dict[str, Any]]) -> str:
        """Convert batch of formatted events to CSV string."""
        if not events:
            return ""

        # Format all events
        formatted_events = self.format_batch(events)

        # Extract data dictionaries
        data_dicts = [event["data"] for event in formatted_events if "data" in event]

        if not data_dicts:
            return ""

        # Get all unique field names
        all_fields = set()
        for data_dict in data_dicts:
            all_fields.update(data_dict.keys())

        # Order fields according to preference
        ordered_fields = []
        for field in self.field_order:
            if field in all_fields:
                ordered_fields.append(field)
                all_fields.remove(field)

        # Add remaining fields
        ordered_fields.extend(sorted(all_fields))

        # Create CSV string
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=ordered_fields)
        writer.writeheader()

        for data_dict in data_dicts:
            # Ensure all fields are present
            row_data = {field: data_dict.get(field, "") for field in ordered_fields}
            writer.writerow(row_data)

        return output.getvalue()

    def _flatten_dict(
        self, data: dict[str, Any], parent_key: str = "", separator: str = "."
    ) -> dict[str, Any]:
        """Flatten nested dictionary for CSV output."""
        items: list[tuple[str, Any]] = []

        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            elif isinstance(value, list):
                # Convert list to comma-separated string
                items.append((new_key, ",".join(str(item) for item in value)))
            else:
                items.append((new_key, value))

        return dict(items)
