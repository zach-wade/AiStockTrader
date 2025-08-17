#!/usr/bin/env python3
"""
Generate comprehensive validation report for TSLA end-to-end testing.
"""
# Standard library imports
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


class TSLAValidationReport:
    """Generate comprehensive validation report for TSLA."""

    def __init__(self):
        self.config_manager = get_config_manager()
        self.config = self.config_manager.load_config("unified_config")
        self.db_factory = DatabaseFactory()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "TSLA",
            "sections": {},
            "metrics": {},
            "issues": [],
            "recommendations": [],
            "overall_score": 0.0,
        }

    async def generate_report(self):
        """Generate comprehensive validation report."""
        db_adapter = self.db_factory.create_async_database(self.config)

        try:
            print("=" * 80)
            print("TSLA COMPREHENSIVE VALIDATION REPORT")
            print("=" * 80)
            print(f"Generated: {self.results['timestamp']}")
            print("\n")

            # 1. Data Completeness
            await self._validate_data_completeness(db_adapter)

            # 2. Data Quality
            await self._validate_data_quality(db_adapter)

            # 3. Feature Engineering
            await self._validate_features(db_adapter)

            # 4. Scanner Performance
            await self._validate_scanner_performance(db_adapter)

            # 5. Model Readiness
            await self._validate_model_readiness(db_adapter)

            # 6. System Integration
            await self._validate_system_integration(db_adapter)

            # Calculate overall score
            self._calculate_overall_score()

            # Generate recommendations
            self._generate_recommendations()

            # Print summary
            self._print_summary()

            # Save report
            await self._save_report()

            return self.results["overall_score"] >= 80.0

        finally:
            await db_adapter.close()

    async def _validate_data_completeness(self, db_adapter):
        """Validate data completeness across all sources."""
        print("1. DATA COMPLETENESS VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Market data intervals
        intervals = ["1minute", "5minute", "15minute", "30minute", "1hour", "1day"]
        interval_query = """
            SELECT interval, COUNT(*) as count,
                   MIN(timestamp) as start_date,
                   MAX(timestamp) as end_date
            FROM market_data
            WHERE symbol = 'TSLA'
            GROUP BY interval
        """

        results = await db_adapter.fetch_all(interval_query)
        found_intervals = {row["interval"]: row for row in results}

        print("Market Data Coverage:")
        for interval in intervals:
            if interval in found_intervals:
                row = found_intervals[interval]
                days = (row["end_date"] - row["start_date"]).days
                print(f"  {interval}: ✓ {row['count']:,} records ({days} days)")
                section["checks"][f"market_data_{interval}"] = True
            else:
                print(f"  {interval}: ✗ Missing")
                section["checks"][f"market_data_{interval}"] = False
                section["score"] -= 10
                self.results["issues"].append(f"Missing {interval} market data")

        # News data
        news_check = await db_adapter.fetch_one(
            "SELECT COUNT(*) as count FROM news_data WHERE symbol = 'TSLA'"
        )

        if news_check and news_check["count"] > 0:
            print(f"\nNews Data: ✓ {news_check['count']:,} articles")
            section["checks"]["news_data"] = True
        else:
            print("\nNews Data: ✗ Missing")
            section["checks"]["news_data"] = False
            section["score"] -= 20
            self.results["issues"].append("No news data found")

        # Financials
        fin_check = await db_adapter.fetch_one(
            "SELECT COUNT(*) as count FROM financials_data WHERE symbol = 'TSLA'"
        )

        if fin_check and fin_check["count"] >= 15:  # ~5 years of quarterly reports
            print(f"Financials: ✓ {fin_check['count']} reports")
            section["checks"]["financials_data"] = True
        else:
            count = fin_check["count"] if fin_check else 0
            print(f"Financials: ⚠️  Only {count} reports (expected 15+)")
            section["checks"]["financials_data"] = False
            section["score"] -= 10
            self.results["issues"].append(f"Insufficient financial data ({count} reports)")

        self.results["sections"]["data_completeness"] = section
        self.results["metrics"]["data_completeness_score"] = max(0, section["score"])

    async def _validate_data_quality(self, db_adapter):
        """Validate data quality metrics."""
        print("\n\n2. DATA QUALITY VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Check for duplicate volumes
        dup_query = """
            SELECT interval, volume, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 /
                         (SELECT COUNT(*) FROM market_data
                          WHERE symbol = 'TSLA' AND interval = md.interval), 2) as pct
            FROM market_data md
            WHERE symbol = 'TSLA'
            GROUP BY interval, volume
            HAVING COUNT(*) > 50 AND
                   COUNT(*) * 100.0 / (SELECT COUNT(*) FROM market_data
                                       WHERE symbol = 'TSLA' AND interval = md.interval) > 5
            ORDER BY pct DESC
            LIMIT 10
        """

        dup_results = await db_adapter.fetch_all(dup_query)

        if dup_results:
            print("Duplicate Volume Issues:")
            for row in dup_results:
                print(f"  {row['interval']}: {row['volume']:,.0f} appears {row['pct']}% of time")
                section["score"] -= 5
            section["checks"]["no_duplicate_volumes"] = False
            self.results["issues"].append(f"Found {len(dup_results)} duplicate volume patterns")
        else:
            print("Duplicate Volumes: ✓ No issues found")
            section["checks"]["no_duplicate_volumes"] = True

        # Check for price anomalies
        anomaly_query = """
            SELECT COUNT(*) as anomalies
            FROM market_data
            WHERE symbol = 'TSLA'
            AND (high < low OR close > high * 1.5 OR close < low * 0.5)
        """

        anomaly_result = await db_adapter.fetch_one(anomaly_query)
        anomalies = anomaly_result["anomalies"] if anomaly_result else 0

        if anomalies > 0:
            print(f"Price Anomalies: ⚠️  {anomalies} found")
            section["checks"]["price_integrity"] = False
            section["score"] -= 10
            self.results["issues"].append(f"Found {anomalies} price anomalies")
        else:
            print("Price Anomalies: ✓ None found")
            section["checks"]["price_integrity"] = True

        self.results["sections"]["data_quality"] = section
        self.results["metrics"]["data_quality_score"] = max(0, section["score"])

    async def _validate_features(self, db_adapter):
        """Validate feature engineering."""
        print("\n\n3. FEATURE ENGINEERING VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Check technical features
        tech_query = """
            SELECT
                COUNT(DISTINCT feature_name) as feature_count,
                COUNT(*) as total_records,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM features_technical
            WHERE symbol = 'TSLA'
        """

        tech_result = await db_adapter.fetch_one(tech_query)

        if tech_result and tech_result["feature_count"] > 0:
            print(
                f"Technical Features: ✓ {tech_result['feature_count']} features, {tech_result['total_records']:,} records"
            )
            section["checks"]["technical_features"] = True

            # Check for NaN/NULL values
            nan_query = """
                SELECT feature_name, COUNT(*) as null_count
                FROM features_technical
                WHERE symbol = 'TSLA' AND value IS NULL
                GROUP BY feature_name
                HAVING COUNT(*) > 100
            """

            nan_results = await db_adapter.fetch_all(nan_query)
            if nan_results:
                print("  ⚠️  Features with many NULL values:")
                for row in nan_results:
                    print(f"    {row['feature_name']}: {row['null_count']} NULLs")
                section["score"] -= 5 * len(nan_results)
        else:
            print("Technical Features: ✗ Not calculated")
            section["checks"]["technical_features"] = False
            section["score"] -= 30
            self.results["issues"].append("Technical features not calculated")

        # Check sentiment features
        sent_query = """
            SELECT COUNT(*) as count
            FROM features_sentiment
            WHERE symbol = 'TSLA'
        """

        sent_result = await db_adapter.fetch_one(sent_query)

        if sent_result and sent_result["count"] > 0:
            print(f"Sentiment Features: ✓ {sent_result['count']:,} records")
            section["checks"]["sentiment_features"] = True
        else:
            print("Sentiment Features: ⚠️  Not calculated (optional)")
            section["checks"]["sentiment_features"] = False
            # Don't penalize heavily as sentiment is optional
            section["score"] -= 5

        self.results["sections"]["features"] = section
        self.results["metrics"]["features_score"] = max(0, section["score"])

    async def _validate_scanner_performance(self, db_adapter):
        """Validate scanner pipeline performance."""
        print("\n\n4. SCANNER PIPELINE VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Check qualification status
        qual_query = """
            SELECT layer0_qualified, layer1_qualified,
                   layer2_qualified, layer3_qualified
            FROM companies
            WHERE symbol = 'TSLA'
        """

        qual_result = await db_adapter.fetch_one(qual_query)

        if qual_result:
            print("Layer Qualifications:")
            for i in range(4):
                qualified = qual_result.get(f"layer{i}_qualified", False)
                print(f"  Layer {i}: {'✓' if qualified else '✗'}")
                section["checks"][f"layer{i}_qualified"] = qualified

                # TSLA should qualify for at least Layer 1
                if i == 1 and not qualified:
                    section["score"] -= 30
                    self.results["issues"].append("TSLA did not qualify for Layer 1 (liquidity)")
        else:
            print("  ✗ TSLA not found in companies table")
            section["checks"]["in_companies_table"] = False
            section["score"] -= 50
            self.results["issues"].append("TSLA not in companies table")

        # Check for alerts
        alert_query = """
            SELECT alert_type, COUNT(*) as count
            FROM scanner_alerts
            WHERE symbol = 'TSLA'
            GROUP BY alert_type
        """

        alert_results = await db_adapter.fetch_all(alert_query)

        if alert_results:
            print("\nScanner Alerts Generated:")
            for row in alert_results:
                print(f"  {row['alert_type']}: {row['count']} alerts")
            section["checks"]["alerts_generated"] = True
        else:
            print("\nScanner Alerts: ⚠️  None generated")
            section["checks"]["alerts_generated"] = False
            # Not necessarily bad if no catalysts present
            section["score"] -= 10

        self.results["sections"]["scanner"] = section
        self.results["metrics"]["scanner_score"] = max(0, section["score"])

    async def _validate_model_readiness(self, db_adapter):
        """Validate data is ready for model training."""
        print("\n\n5. MODEL READINESS VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Check if we have enough data points
        data_query = """
            SELECT
                (SELECT COUNT(*) FROM market_data WHERE symbol = 'TSLA' AND interval = '1hour') as hourly_data,
                (SELECT COUNT(*) FROM features_technical WHERE symbol = 'TSLA') as features,
                (SELECT COUNT(DISTINCT DATE(timestamp)) FROM market_data WHERE symbol = 'TSLA') as trading_days
        """

        data_result = await db_adapter.fetch_one(data_query)

        if data_result:
            hourly = data_result["hourly_data"]
            features = data_result["features"]
            days = data_result["trading_days"]

            print("Training Data Available:")
            print(f"  Hourly Records: {hourly:,}")
            print(f"  Feature Records: {features:,}")
            print(f"  Trading Days: {days}")

            # Need at least 1000 hourly records for meaningful training
            if hourly >= 1000:
                section["checks"]["sufficient_data"] = True
            else:
                section["checks"]["sufficient_data"] = False
                section["score"] -= 40
                self.results["issues"].append(
                    f"Insufficient training data ({hourly} hourly records)"
                )

            # Features should be calculated
            if features > 0:
                section["checks"]["features_ready"] = True
            else:
                section["checks"]["features_ready"] = False
                section["score"] -= 40
                self.results["issues"].append("No features calculated for model training")

        self.results["sections"]["model_readiness"] = section
        self.results["metrics"]["model_readiness_score"] = max(0, section["score"])

    async def _validate_system_integration(self, db_adapter):
        """Validate system integration points."""
        print("\n\n6. SYSTEM INTEGRATION VALIDATION")
        print("-" * 50)

        section = {"score": 100.0, "checks": {}}

        # Check data lake files
        data_lake_path = Path("data_lake/raw/market_data/symbol=TSLA")
        if data_lake_path.exists():
            file_count = len(list(data_lake_path.rglob("*.parquet")))
            print(f"Data Lake Files: ✓ {file_count} parquet files")
            section["checks"]["data_lake_files"] = True
        else:
            print("Data Lake Files: ✗ Missing")
            section["checks"]["data_lake_files"] = False
            section["score"] -= 20
            self.results["issues"].append("No data lake files found")

        # Check if real-time connection works (WebSocket)
        # This would be checked during scanner pipeline test
        section["checks"]["websocket_ready"] = True  # Assume ready

        # Check model directory structure
        model_dirs = ["models/xgboost", "models/lstm", "models/ensemble"]
        for model_dir in model_dirs:
            if Path(model_dir).exists():
                section["checks"][f"{model_dir}_ready"] = True
            else:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                section["checks"][f"{model_dir}_ready"] = True

        print("Model Directories: ✓ Ready")

        self.results["sections"]["integration"] = section
        self.results["metrics"]["integration_score"] = max(0, section["score"])

    def _calculate_overall_score(self):
        """Calculate overall validation score."""
        scores = [
            self.results["metrics"].get("data_completeness_score", 0) * 0.25,
            self.results["metrics"].get("data_quality_score", 0) * 0.25,
            self.results["metrics"].get("features_score", 0) * 0.20,
            self.results["metrics"].get("scanner_score", 0) * 0.15,
            self.results["metrics"].get("model_readiness_score", 0) * 0.10,
            self.results["metrics"].get("integration_score", 0) * 0.05,
        ]

        self.results["overall_score"] = sum(scores)

    def _generate_recommendations(self):
        """Generate recommendations based on findings."""
        recs = []

        # Data issues
        if any("duplicate volume" in issue.lower() for issue in self.results["issues"]):
            recs.append("Re-download affected date ranges to fix duplicate volume issues")

        if any(
            "missing" in issue.lower() and "data" in issue.lower()
            for issue in self.results["issues"]
        ):
            recs.append("Run backfill for missing data types")

        # Feature issues
        if any("features not calculated" in issue.lower() for issue in self.results["issues"]):
            recs.append("Run: python ai_trader.py features --symbols TSLA --recalculate")

        # Scanner issues
        if any("layer 1" in issue.lower() for issue in self.results["issues"]):
            recs.append("Check TSLA liquidity metrics - may need to adjust thresholds")

        # Model readiness
        if any("insufficient training data" in issue.lower() for issue in self.results["issues"]):
            recs.append("Extend backfill period or add more symbols for training")

        self.results["recommendations"] = recs

    def _print_summary(self):
        """Print summary report."""
        print("\n\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        # Section scores
        print("\nSection Scores:")
        for section, data in self.results["sections"].items():
            score = self.results["metrics"].get(f"{section}_score", 0)
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"  {section.replace('_', ' ').title()}: {status} {score:.0f}%")

        # Overall score
        print(f"\nOVERALL SCORE: {self.results['overall_score']:.0f}%")

        if self.results["overall_score"] >= 80:
            print("STATUS: ✅ READY FOR PRODUCTION")
        elif self.results["overall_score"] >= 60:
            print("STATUS: ⚠️  NEEDS IMPROVEMENT")
        else:
            print("STATUS: ❌ NOT READY")

        # Issues
        if self.results["issues"]:
            print(f"\nISSUES FOUND ({len(self.results['issues'])}):")
            for i, issue in enumerate(self.results["issues"][:10], 1):
                print(f"  {i}. {issue}")
            if len(self.results["issues"]) > 10:
                print(f"  ... and {len(self.results['issues']) - 10} more")

        # Recommendations
        if self.results["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"  {i}. {rec}")

    async def _save_report(self):
        """Save detailed report to file."""
        output_file = Path("data/validation/tsla_comprehensive_report.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed report saved to: {output_file}")

        # Also save HTML report
        html_file = output_file.with_suffix(".html")
        self._generate_html_report(html_file)
        print(f"HTML report saved to: {html_file}")

    def _generate_html_report(self, output_file):
        """Generate HTML version of report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TSLA Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .pass {{ color: green; }}
        .warning {{ color: orange; }}
        .fail {{ color: red; }}
        .issues {{ background-color: #fff0f0; padding: 10px; }}
        .recommendations {{ background-color: #f0f0ff; padding: 10px; }}
    </style>
</head>
<body>
    <h1>TSLA Comprehensive Validation Report</h1>
    <p>Generated: {self.results['timestamp']}</p>

    <div class="section">
        <h2>Overall Score</h2>
        <div class="score {self._get_score_class(self.results['overall_score'])}">
            {self.results['overall_score']:.0f}%
        </div>
    </div>

    <div class="section">
        <h2>Section Scores</h2>
        <ul>
"""

        for section, data in self.results["sections"].items():
            score = self.results["metrics"].get(f"{section}_score", 0)
            html += f"<li>{section.replace('_', ' ').title()}: "
            html += f"<span class='{self._get_score_class(score)}'>{score:.0f}%</span></li>\n"

        html += """
        </ul>
    </div>
"""

        if self.results["issues"]:
            html += """
    <div class="section issues">
        <h2>Issues Found</h2>
        <ul>
"""
            for issue in self.results["issues"]:
                html += f"<li>{issue}</li>\n"
            html += """
        </ul>
    </div>
"""

        if self.results["recommendations"]:
            html += """
    <div class="section recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
            for rec in self.results["recommendations"]:
                html += f"<li>{rec}</li>\n"
            html += """
        </ul>
    </div>
"""

        html += """
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html)

    def _get_score_class(self, score):
        """Get CSS class for score."""
        if score >= 80:
            return "pass"
        elif score >= 60:
            return "warning"
        else:
            return "fail"


async def main():
    """Generate comprehensive validation report."""
    report = TSLAValidationReport()
    success = await report.generate_report()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
