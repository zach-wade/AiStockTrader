"""
Trading Universe Analysis

Analysis tools for comparing and tracking trading universes.
"""

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any

# Third-party imports
import numpy as np

logger = logging.getLogger(__name__)


class UniverseAnalyzer:
    """Tools for analyzing trading universes."""

    def __init__(self, manager):
        """Initialize analyzer with universe manager."""
        self.manager = manager

    def compare_universes(
        self, name1: str, name2: str, reference_date: datetime | None = None
    ) -> dict[str, Any]:
        """Compare two universes."""
        symbols1 = self.manager.get_universe_symbols(name1, reference_date)
        symbols2 = self.manager.get_universe_symbols(name2, reference_date)

        intersection = symbols1 & symbols2
        union = symbols1 | symbols2

        return {
            "universe1": name1,
            "universe2": name2,
            "symbols1_count": len(symbols1),
            "symbols2_count": len(symbols2),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard_similarity": len(intersection) / len(union) if union else 0,
            "overlap_percentage": (
                len(intersection) / min(len(symbols1), len(symbols2)) * 100
                if min(len(symbols1), len(symbols2)) > 0
                else 0
            ),
            "only_in_universe1": symbols1 - symbols2,
            "only_in_universe2": symbols2 - symbols1,
            "common_symbols": intersection,
        }

    def track_universe_changes(self, name: str, lookback_days: int = 30) -> dict[str, Any]:
        """Track changes in universe over time."""
        if name not in self.manager.universe_history:
            return {"error": f"Universe {name} not found"}

        history = self.manager.universe_history[name]
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        recent_history = [s for s in history if s.timestamp >= cutoff_date]

        if len(recent_history) < 2:
            return {"error": "Insufficient history for comparison"}

        changes = []
        for i in range(1, len(recent_history)):
            prev_snapshot = recent_history[i - 1]
            curr_snapshot = recent_history[i]

            added = curr_snapshot.symbols - prev_snapshot.symbols
            removed = prev_snapshot.symbols - curr_snapshot.symbols

            if added or removed:
                changes.append(
                    {
                        "date": curr_snapshot.timestamp.isoformat(),
                        "added": list(added),
                        "removed": list(removed),
                        "added_count": len(added),
                        "removed_count": len(removed),
                        "total_symbols": len(curr_snapshot.symbols),
                        "turnover_rate": (
                            (len(added) + len(removed)) / len(prev_snapshot.symbols) * 100
                            if prev_snapshot.symbols
                            else 0
                        ),
                    }
                )

        # Calculate summary statistics
        total_additions = sum(c["added_count"] for c in changes)
        total_removals = sum(c["removed_count"] for c in changes)
        avg_turnover = np.mean([c["turnover_rate"] for c in changes]) if changes else 0

        return {
            "universe": name,
            "lookback_days": lookback_days,
            "total_changes": len(changes),
            "total_additions": total_additions,
            "total_removals": total_removals,
            "avg_turnover_rate": avg_turnover,
            "changes": changes,
        }

    def get_universe_statistics(self, name: str) -> dict[str, Any]:
        """Get comprehensive statistics for a universe."""
        if name not in self.manager.universe_history:
            return {"error": f"Universe {name} not found"}

        history = self.manager.universe_history[name]
        if not history:
            return {"error": "No history available"}

        latest = history[-1]

        stats = {
            "name": name,
            "current_symbol_count": len(latest.symbols),
            "first_snapshot": history[0].timestamp.isoformat(),
            "last_snapshot": latest.timestamp.isoformat(),
            "total_snapshots": len(history),
            "metadata": latest.metadata,
        }

        # Calculate stability metrics
        if len(history) > 1:
            symbol_counts = [len(s.symbols) for s in history]
            stats["avg_symbol_count"] = np.mean(symbol_counts)
            stats["min_symbol_count"] = min(symbol_counts)
            stats["max_symbol_count"] = max(symbol_counts)
            stats["symbol_count_std"] = np.std(symbol_counts)
            stats["symbol_count_cv"] = (
                stats["symbol_count_std"] / stats["avg_symbol_count"]
                if stats["avg_symbol_count"] > 0
                else 0
            )

            # Calculate turnover
            recent_snapshots = history[-10:]  # Last 10 snapshots
            if len(recent_snapshots) > 1:
                turnovers = []
                for i in range(1, len(recent_snapshots)):
                    prev_symbols = recent_snapshots[i - 1].symbols
                    curr_symbols = recent_snapshots[i].symbols

                    if prev_symbols:
                        turnover = len(prev_symbols - curr_symbols) / len(prev_symbols)
                        turnovers.append(turnover)

                if turnovers:
                    stats["avg_turnover"] = np.mean(turnovers)
                    stats["max_turnover"] = max(turnovers)
                    stats["min_turnover"] = min(turnovers)
                    stats["turnover_std"] = np.std(turnovers)

        return stats

    def analyze_universe_stability(self, name: str, window_days: int = 30) -> dict[str, Any]:
        """Analyze universe stability over time."""
        if name not in self.manager.universe_history:
            return {"error": f"Universe {name} not found"}

        history = self.manager.universe_history[name]
        if len(history) < 2:
            return {"error": "Insufficient history for stability analysis"}

        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_history = [s for s in history if s.timestamp >= cutoff_date]

        if len(recent_history) < 2:
            return {"error": "Insufficient recent history for stability analysis"}

        # Calculate stability metrics
        stability_scores = []
        for i in range(1, len(recent_history)):
            prev_symbols = recent_history[i - 1].symbols
            curr_symbols = recent_history[i].symbols

            if prev_symbols and curr_symbols:
                # Jaccard similarity as stability score
                intersection = prev_symbols & curr_symbols
                union = prev_symbols | curr_symbols
                stability = len(intersection) / len(union) if union else 0
                stability_scores.append(stability)

        if not stability_scores:
            return {"error": "Could not calculate stability scores"}

        return {
            "universe": name,
            "window_days": window_days,
            "avg_stability": np.mean(stability_scores),
            "min_stability": min(stability_scores),
            "max_stability": max(stability_scores),
            "stability_std": np.std(stability_scores),
            "stability_trend": (
                "increasing"
                if stability_scores[-1] > stability_scores[0]
                else "decreasing" if stability_scores[-1] < stability_scores[0] else "stable"
            ),
            "snapshots_analyzed": len(recent_history),
        }

    def get_universe_overlap_matrix(self, universe_names: list[str]) -> dict[str, Any]:
        """Calculate overlap matrix between multiple universes."""
        if len(universe_names) < 2:
            return {"error": "At least 2 universes required for overlap matrix"}

        # Get symbols for all universes
        universe_symbols = {}
        for name in universe_names:
            symbols = self.manager.get_universe_symbols(name)
            if symbols:
                universe_symbols[name] = symbols

        if len(universe_symbols) < 2:
            return {"error": "At least 2 universes with symbols required"}

        # Calculate overlap matrix
        overlap_matrix = {}
        jaccard_matrix = {}

        for name1 in universe_symbols:
            overlap_matrix[name1] = {}
            jaccard_matrix[name1] = {}

            for name2 in universe_symbols:
                if name1 == name2:
                    overlap_matrix[name1][name2] = len(universe_symbols[name1])
                    jaccard_matrix[name1][name2] = 1.0
                else:
                    intersection = universe_symbols[name1] & universe_symbols[name2]
                    union = universe_symbols[name1] | universe_symbols[name2]

                    overlap_matrix[name1][name2] = len(intersection)
                    jaccard_matrix[name1][name2] = len(intersection) / len(union) if union else 0

        return {
            "universes": list(universe_symbols.keys()),
            "overlap_matrix": overlap_matrix,
            "jaccard_matrix": jaccard_matrix,
            "universe_sizes": {name: len(symbols) for name, symbols in universe_symbols.items()},
        }

    def find_unique_symbols(
        self, target_universe: str, comparison_universes: list[str]
    ) -> dict[str, Any]:
        """Find symbols unique to target universe compared to others."""
        target_symbols = self.manager.get_universe_symbols(target_universe)

        if not target_symbols:
            return {"error": f"Target universe {target_universe} has no symbols"}

        # Get all symbols from comparison universes
        comparison_symbols = set()
        for name in comparison_universes:
            comparison_symbols.update(self.manager.get_universe_symbols(name))

        unique_symbols = target_symbols - comparison_symbols

        return {
            "target_universe": target_universe,
            "comparison_universes": comparison_universes,
            "target_symbol_count": len(target_symbols),
            "comparison_symbol_count": len(comparison_symbols),
            "unique_symbols": list(unique_symbols),
            "unique_symbol_count": len(unique_symbols),
            "uniqueness_ratio": len(unique_symbols) / len(target_symbols) if target_symbols else 0,
        }

    def get_sector_analysis(self, name: str) -> dict[str, Any]:
        """Get detailed sector analysis for a universe."""
        distribution = self.manager.get_sector_distribution(name)

        if not distribution:
            return {"error": f"No sector data available for universe {name}"}

        total_symbols = sum(distribution.values())

        # Calculate sector percentages
        sector_percentages = {
            sector: (count / total_symbols) * 100 for sector, count in distribution.items()
        }

        # Find dominant and minor sectors
        sorted_sectors = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        return {
            "universe": name,
            "total_symbols": total_symbols,
            "sector_count": len(distribution),
            "sector_distribution": distribution,
            "sector_percentages": sector_percentages,
            "dominant_sector": sorted_sectors[0][0] if sorted_sectors else None,
            "dominant_sector_percentage": (
                sector_percentages.get(sorted_sectors[0][0], 0) if sorted_sectors else 0
            ),
            "minor_sectors": [
                sector for sector, count in sorted_sectors if count < total_symbols * 0.05
            ],  # < 5%
            "concentration_score": max(sector_percentages.values()) if sector_percentages else 0,
        }
