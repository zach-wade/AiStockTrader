"""
Outcome metrics calculation for classification models.

This module calculates comprehensive metrics for outcome classification including:
- Classification performance metrics
- Trading-specific metrics
- Risk-adjusted returns
- Statistical significance tests
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy import stats

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Classification performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    
    # Per-class metrics
    class_precision: Dict[str, float] = field(default_factory=dict)
    class_recall: Dict[str, float] = field(default_factory=dict)
    class_f1: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    
    # Sample counts
    total_samples: int = 0
    support_per_class: Dict[str, int] = field(default_factory=dict)


@dataclass
class TradingMetrics:
    """Trading-specific outcome metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Win/loss statistics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    calmar_ratio: float
    
    # Transaction costs
    total_trades: int
    transaction_costs: float
    net_return: float


@dataclass
class StatisticalTests:
    """Statistical significance tests."""
    # Accuracy tests
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    accuracy_p_value: float
    
    # Return tests
    returns_t_stat: float
    returns_p_value: float
    returns_significant: bool
    
    # Distribution tests
    normality_p_value: float
    returns_skewness: float
    returns_kurtosis: float
    
    # Baseline comparison
    vs_random_p_value: Optional[float] = None
    vs_benchmark_p_value: Optional[float] = None


@dataclass
class OutcomeMetricsResult:
    """Comprehensive outcome metrics result."""
    model_name: str
    evaluation_period: Tuple[datetime, datetime]
    horizon_hours: int
    
    # Core metrics
    classification_metrics: ClassificationMetrics
    trading_metrics: TradingMetrics
    statistical_tests: StatisticalTests
    
    # Data quality
    data_quality_score: float
    sample_size_adequacy: bool
    class_balance_ratio: float
    
    # Feature importance (if available)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: List[str] = field(default_factory=list)


class OutcomeMetricsCalculator(ErrorHandlingMixin):
    """
    Calculates comprehensive metrics for outcome classification models.
    
    Features:
    - Classification performance metrics
    - Trading-specific performance analysis
    - Statistical significance testing
    - Risk-adjusted return calculations
    - Benchmark comparisons
    """
    
    def __init__(self, db_pool: DatabasePool, risk_free_rate: float = 0.02):
        """Initialize outcome metrics calculator."""
        super().__init__()
        self.db_pool = db_pool
        self.risk_free_rate = risk_free_rate
        
        # Trading assumptions
        self._transaction_cost_bps = 5  # 5 basis points per trade
        self._position_size = 1.0  # Normalized position size
        self._trading_days_per_year = 252
        
        # Statistical test parameters
        self._confidence_level = 0.95
        self._min_sample_size = 100
        
    @timer
    async def calculate_comprehensive_metrics(
        self,
        model_name: str,
        y_true: List[str],
        y_pred: List[str],
        y_proba: Optional[List[List[float]]] = None,
        returns: Optional[List[float]] = None,
        evaluation_period: Optional[Tuple[datetime, datetime]] = None,
        horizon_hours: int = 24
    ) -> OutcomeMetricsResult:
        """
        Calculate comprehensive outcome metrics.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            returns: Actual returns for trading metrics
            evaluation_period: Evaluation period
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Comprehensive metrics result
        """
        with self._handle_error("calculating comprehensive metrics"):
            # Validate inputs
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have same length")
            
            if returns and len(returns) != len(y_true):
                raise ValueError("returns must have same length as labels")
            
            # Calculate classification metrics
            classification_metrics = self._calculate_classification_metrics(
                y_true, y_pred, y_proba
            )
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(
                y_true, y_pred, returns or [0.0] * len(y_true)
            )
            
            # Calculate statistical tests
            statistical_tests = self._calculate_statistical_tests(
                y_true, y_pred, returns or [0.0] * len(y_true)
            )
            
            # Assess data quality
            data_quality_score = self._assess_data_quality(
                y_true, y_pred, returns
            )
            
            # Check sample size adequacy
            sample_size_adequacy = len(y_true) >= self._min_sample_size
            
            # Calculate class balance
            class_counts = pd.Series(y_true).value_counts()
            class_balance_ratio = class_counts.min() / class_counts.max()
            
            # Create result
            result = OutcomeMetricsResult(
                model_name=model_name,
                evaluation_period=evaluation_period or (
                    datetime.utcnow() - timedelta(days=30),
                    datetime.utcnow()
                ),
                horizon_hours=horizon_hours,
                classification_metrics=classification_metrics,
                trading_metrics=trading_metrics,
                statistical_tests=statistical_tests,
                data_quality_score=data_quality_score,
                sample_size_adequacy=sample_size_adequacy,
                class_balance_ratio=class_balance_ratio
            )
            
            # Add quality notes
            result.notes = self._generate_quality_notes(result)
            
            # Record metrics
            record_metric(
                'outcome_metrics_calculator.metrics_calculated',
                1,
                tags={
                    'model': model_name,
                    'samples': len(y_true),
                    'accuracy': classification_metrics.accuracy
                }
            )
            
            logger.info(
                f"Calculated comprehensive metrics for {model_name}: "
                f"accuracy={classification_metrics.accuracy:.3f}, "
                f"sharpe={trading_metrics.sharpe_ratio:.3f}"
            )
            
            return result
    
    def _calculate_classification_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_proba: Optional[List[List[float]]] = None
    ) -> ClassificationMetrics:
        """Calculate classification performance metrics."""
        # Convert to numpy arrays
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        # Get unique classes
        classes = sorted(list(set(y_true + y_pred)))
        
        # Basic metrics
        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        precision = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        recall = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        support_per_class = {}
        
        try:
            per_class_precision = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            per_class_recall = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            per_class_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            
            for i, class_name in enumerate(classes):
                if i < len(per_class_precision):
                    class_precision[class_name] = per_class_precision[i]
                    class_recall[class_name] = per_class_recall[i]
                    class_f1[class_name] = per_class_f1[i]
                    support_per_class[class_name] = int(np.sum(y_true_arr == class_name))
        except Exception as e:
            logger.warning(f"Error calculating per-class metrics: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
        
        # AUC score (for binary classification)
        auc_score = None
        if len(classes) == 2 and y_proba is not None:
            try:
                # Convert string labels to binary
                y_true_binary = (y_true_arr == classes[1]).astype(int)
                y_proba_pos = np.array(y_proba)[:, 1] if len(y_proba[0]) == 2 else None
                
                if y_proba_pos is not None:
                    auc_score = roc_auc_score(y_true_binary, y_proba_pos)
            except Exception as e:
                logger.warning(f"Error calculating AUC: {e}")
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            class_precision=class_precision,
            class_recall=class_recall,
            class_f1=class_f1,
            confusion_matrix=cm,
            class_names=classes,
            total_samples=len(y_true),
            support_per_class=support_per_class
        )
    
    def _calculate_trading_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        returns: List[float]
    ) -> TradingMetrics:
        """Calculate trading-specific metrics."""
        # Convert predictions to trading signals
        signals = self._predictions_to_signals(y_pred)
        
        # Calculate strategy returns
        strategy_returns = np.array(signals) * np.array(returns)
        
        # Remove transaction costs
        transaction_costs = self._calculate_transaction_costs(signals)
        net_returns = strategy_returns - transaction_costs
        
        # Basic return metrics
        total_return = np.sum(net_returns)
        mean_return = np.mean(net_returns)
        return_std = np.std(net_returns)
        
        # Annualized metrics
        annualized_return = mean_return * self._trading_days_per_year
        volatility = return_std * np.sqrt(self._trading_days_per_year)
        
        # Sharpe ratio
        sharpe_ratio = (
            (annualized_return - self.risk_free_rate) / volatility
            if volatility > 0 else 0.0
        )
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(net_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win/loss statistics
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]
        
        win_rate = len(winning_trades) / len(net_returns) if len(net_returns) > 0 else 0.0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        profit_factor = (
            np.sum(winning_trades) / abs(np.sum(losing_trades))
            if len(losing_trades) > 0 and np.sum(losing_trades) != 0 else 0.0
        )
        
        # Risk metrics
        var_95 = np.percentile(net_returns, 5) if len(net_returns) > 0 else 0.0
        cvar_95 = np.mean(net_returns[net_returns <= var_95]) if len(net_returns) > 0 else 0.0
        
        calmar_ratio = (
            annualized_return / abs(max_drawdown)
            if max_drawdown != 0 else 0.0
        )
        
        return TradingMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            total_trades=len([s for s in signals if s != 0]),
            transaction_costs=np.sum(transaction_costs),
            net_return=total_return
        )
    
    def _calculate_statistical_tests(
        self,
        y_true: List[str],
        y_pred: List[str],
        returns: List[float]
    ) -> StatisticalTests:
        """Calculate statistical significance tests."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        returns_arr = np.array(returns)
        
        # Accuracy confidence interval
        n = len(y_true)
        accuracy = np.mean(y_true_arr == y_pred_arr)
        
        # Wilson score interval for binomial proportion
        z_score = stats.norm.ppf(1 - (1 - self._confidence_level) / 2)
        
        p_hat = accuracy
        n_eff = n + z_score**2
        p_tilde = (n * p_hat + z_score**2 / 2) / n_eff
        margin = z_score * np.sqrt(p_tilde * (1 - p_tilde) / n_eff)
        
        accuracy_ci_lower = max(0, p_tilde - margin)
        accuracy_ci_upper = min(1, p_tilde + margin)
        
        # Test if accuracy is significantly better than random
        # Null hypothesis: accuracy = 1 / num_classes
        num_classes = len(set(y_true))
        null_accuracy = 1 / num_classes
        
        z_stat = (accuracy - null_accuracy) / np.sqrt(null_accuracy * (1 - null_accuracy) / n)
        accuracy_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Returns t-test (test if mean return is significantly different from 0)
        if len(returns_arr) > 1:
            returns_t_stat, returns_p_value = stats.ttest_1samp(returns_arr, 0)
            returns_significant = returns_p_value < (1 - self._confidence_level)
        else:
            returns_t_stat = 0.0
            returns_p_value = 1.0
            returns_significant = False
        
        # Normality test for returns
        if len(returns_arr) > 8:  # Minimum for Shapiro-Wilk test
            _, normality_p_value = stats.shapiro(returns_arr)
        else:
            normality_p_value = 1.0
        
        # Return distribution moments
        returns_skewness = float(stats.skew(returns_arr)) if len(returns_arr) > 0 else 0.0
        returns_kurtosis = float(stats.kurtosis(returns_arr)) if len(returns_arr) > 0 else 0.0
        
        return StatisticalTests(
            accuracy_ci_lower=accuracy_ci_lower,
            accuracy_ci_upper=accuracy_ci_upper,
            accuracy_p_value=accuracy_p_value,
            returns_t_stat=float(returns_t_stat),
            returns_p_value=float(returns_p_value),
            returns_significant=returns_significant,
            normality_p_value=float(normality_p_value),
            returns_skewness=returns_skewness,
            returns_kurtosis=returns_kurtosis
        )
    
    def _predictions_to_signals(
        self,
        y_pred: List[str]
    ) -> List[float]:
        """Convert predictions to trading signals."""
        signals = []
        
        for pred in y_pred:
            if pred in ['up', 'buy', 'long', 'positive']:
                signals.append(1.0)
            elif pred in ['down', 'sell', 'short', 'negative']:
                signals.append(-1.0)
            else:  # flat, hold, neutral
                signals.append(0.0)
        
        return signals
    
    def _calculate_transaction_costs(
        self,
        signals: List[float]
    ) -> np.ndarray:
        """Calculate transaction costs based on signal changes."""
        signals_arr = np.array(signals)
        
        # Calculate position changes
        prev_position = 0.0
        costs = []
        
        for signal in signals_arr:
            position_change = abs(signal - prev_position)
            cost = position_change * self._transaction_cost_bps / 10000  # Convert bps to decimal
            costs.append(cost)
            prev_position = signal
        
        return np.array(costs)
    
    def _assess_data_quality(
        self,
        y_true: List[str],
        y_pred: List[str],
        returns: Optional[List[float]]
    ) -> float:
        """Assess data quality score."""
        score = 1.0
        
        # Check sample size
        if len(y_true) < self._min_sample_size:
            score -= 0.3
        
        # Check class balance
        class_counts = pd.Series(y_true).value_counts()
        min_class_ratio = class_counts.min() / class_counts.sum()
        if min_class_ratio < 0.1:  # Less than 10% for smallest class
            score -= 0.2
        
        # Check for missing returns
        if returns is None:
            score -= 0.2
        elif returns and any(r is None or np.isnan(r) for r in returns):
            score -= 0.1
        
        # Check prediction diversity
        pred_diversity = len(set(y_pred)) / len(set(y_true))
        if pred_diversity < 0.5:
            score -= 0.2
        
        return max(0.0, score)
    
    def _generate_quality_notes(
        self,
        result: OutcomeMetricsResult
    ) -> List[str]:
        """Generate quality assessment notes."""
        notes = []
        
        # Sample size
        if not result.sample_size_adequacy:
            notes.append(
                f"Small sample size ({result.classification_metrics.total_samples}). "
                f"Results may not be reliable."
            )
        
        # Class imbalance
        if result.class_balance_ratio < 0.3:
            notes.append(
                f"Significant class imbalance detected (ratio: {result.class_balance_ratio:.2f}). "
                f"Consider using stratified sampling or class weights."
            )
        
        # Statistical significance
        if not result.statistical_tests.returns_significant:
            notes.append(
                "Returns are not statistically significant. "
                "Model may not have predictive power."
            )
        
        # Low accuracy
        if result.classification_metrics.accuracy < 0.6:
            notes.append(
                f"Low accuracy ({result.classification_metrics.accuracy:.2%}). "
                f"Model performance may be inadequate."
            )
        
        # High volatility
        if result.trading_metrics.volatility > 0.3:  # 30% annualized volatility
            notes.append(
                f"High volatility ({result.trading_metrics.volatility:.1%}). "
                f"Consider risk management measures."
            )
        
        # Poor Sharpe ratio
        if result.trading_metrics.sharpe_ratio < 0.5:
            notes.append(
                f"Low Sharpe ratio ({result.trading_metrics.sharpe_ratio:.2f}). "
                f"Risk-adjusted returns may be insufficient."
            )
        
        return notes
    
    async def compare_models(
        self,
        results: List[OutcomeMetricsResult]
    ) -> Dict[str, Any]:
        """Compare multiple model results."""
        if len(results) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        comparison = {
            "models": [r.model_name for r in results],
            "metrics_comparison": {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            ("accuracy", lambda r: r.classification_metrics.accuracy),
            ("f1_score", lambda r: r.classification_metrics.f1_score),
            ("sharpe_ratio", lambda r: r.trading_metrics.sharpe_ratio),
            ("max_drawdown", lambda r: r.trading_metrics.max_drawdown),
            ("win_rate", lambda r: r.trading_metrics.win_rate),
            ("total_return", lambda r: r.trading_metrics.total_return)
        ]
        
        for metric_name, metric_func in metrics_to_compare:
            values = [metric_func(r) for r in results]
            
            comparison["metrics_comparison"][metric_name] = {
                "values": dict(zip([r.model_name for r in results], values)),
                "best_model": results[np.argmax(values) if metric_name != "max_drawdown" 
                                    else np.argmin(np.abs(values))].model_name,
                "range": [min(values), max(values)],
                "std": float(np.std(values))
            }
        
        # Statistical tests for significant differences
        if len(results) == 2:
            # Simple pairwise comparison
            model_a, model_b = results[0], results[1]
            
            comparison["statistical_comparison"] = {
                "accuracy_difference": (
                    model_b.classification_metrics.accuracy - 
                    model_a.classification_metrics.accuracy
                ),
                "sharpe_difference": (
                    model_b.trading_metrics.sharpe_ratio - 
                    model_a.trading_metrics.sharpe_ratio
                )
            }
        
        return comparison
    
    def export_metrics_to_dict(
        self,
        result: OutcomeMetricsResult
    ) -> Dict[str, Any]:
        """Export metrics result to dictionary format."""
        return {
            "model_name": result.model_name,
            "evaluation_period": [
                result.evaluation_period[0].isoformat(),
                result.evaluation_period[1].isoformat()
            ],
            "horizon_hours": result.horizon_hours,
            "classification_metrics": {
                "accuracy": result.classification_metrics.accuracy,
                "precision": result.classification_metrics.precision,
                "recall": result.classification_metrics.recall,
                "f1_score": result.classification_metrics.f1_score,
                "auc_score": result.classification_metrics.auc_score,
                "total_samples": result.classification_metrics.total_samples
            },
            "trading_metrics": {
                "total_return": result.trading_metrics.total_return,
                "annualized_return": result.trading_metrics.annualized_return,
                "volatility": result.trading_metrics.volatility,
                "sharpe_ratio": result.trading_metrics.sharpe_ratio,
                "max_drawdown": result.trading_metrics.max_drawdown,
                "win_rate": result.trading_metrics.win_rate,
                "profit_factor": result.trading_metrics.profit_factor
            },
            "statistical_tests": {
                "accuracy_ci": [
                    result.statistical_tests.accuracy_ci_lower,
                    result.statistical_tests.accuracy_ci_upper
                ],
                "returns_significant": result.statistical_tests.returns_significant,
                "returns_p_value": result.statistical_tests.returns_p_value
            },
            "data_quality": {
                "data_quality_score": result.data_quality_score,
                "sample_size_adequacy": result.sample_size_adequacy,
                "class_balance_ratio": result.class_balance_ratio
            },
            "notes": result.notes,
            "calculation_timestamp": result.calculation_timestamp.isoformat()
        }
    
    def configure_trading_parameters(
        self,
        transaction_cost_bps: float = 5,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252
    ) -> None:
        """Configure trading calculation parameters."""
        self._transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate
        self._trading_days_per_year = trading_days_per_year
        
        logger.info(
            f"Updated trading parameters: transaction_cost={transaction_cost_bps}bps, "
            f"risk_free_rate={risk_free_rate:.1%}"
        )