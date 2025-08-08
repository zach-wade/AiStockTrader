"""
Nonlinear Dynamics Calculator

Specialized calculator for chaos theory and nonlinear dynamics analysis including:
- Lyapunov exponent estimation (multiple methods)
- Correlation dimension calculation
- Recurrence quantification analysis (RQA)
- Phase space reconstruction
- Chaos detection (0-1 test)
- Attractor analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from .base_statistical import BaseStatisticalCalculator

logger = logging.getLogger(__name__)


class NonlinearCalculator(BaseStatisticalCalculator):
    """Calculator for nonlinear dynamics and chaos theory measures."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize nonlinear dynamics calculator."""
        super().__init__(config)
        
        # Phase space reconstruction parameters
        self.embedding_dimension = self.stat_config.embedding_dimension
        self.time_delay = self.config.get('time_delay', 1)
        self.max_embedding_dim = self.config.get('max_embedding_dim', 10)
        
        # Lyapunov exponent parameters
        self.lyapunov_k = self.config.get('lyapunov_k', 5)  # Evolution time
        self.lyapunov_min_neighbors = self.config.get('lyapunov_min_neighbors', 10)
        
        # Correlation dimension parameters
        self.correlation_r_min = self.config.get('correlation_r_min', 0.01)
        self.correlation_r_max = self.stat_config.correlation_dimension_r_max
        self.correlation_n_radii = self.config.get('correlation_n_radii', 15)
        
        # Recurrence analysis parameters
        self.recurrence_threshold = self.stat_config.recurrence_threshold
        self.recurrence_min_line_length = self.config.get('recurrence_min_line_length', 2)
        
        # Minimum data requirements
        self.min_data_for_nonlinear = self.config.get('min_data_for_nonlinear', 100)
    
    def get_feature_names(self) -> List[str]:
        """Return list of nonlinear dynamics feature names."""
        feature_names = []
        
        # Lyapunov exponents
        feature_names.extend([
            'lyapunov_largest', 'lyapunov_wolf', 'lyapunov_rosenstein', 'lyapunov_kantz'
        ])
        
        # Correlation dimension
        feature_names.append('correlation_dimension')
        
        # Recurrence quantification analysis
        feature_names.extend([
            'recurrence_rate', 'determinism', 'laminarity', 'entropy_rqa',
            'max_line_length', 'divergence', 'trend_rqa'
        ])
        
        # Chaos detection
        feature_names.extend([
            'chaos_test_01', 'embedding_dimension_opt', 'false_nearest_neighbors'
        ])
        
        # Attractor analysis
        feature_names.extend([
            'attractor_dimension', 'predictability', 'complexity_lmc'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate nonlinear dynamics features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with nonlinear dynamics features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)
            
            # Calculate returns for nonlinear analysis
            returns = self.calculate_returns(data['close'])
            
            # Use sufficient window for nonlinear analysis
            nonlinear_window = max(self.min_data_for_nonlinear, 150)
            
            # Calculate Lyapunov exponents
            features = self._calculate_lyapunov_features(returns, features, nonlinear_window)
            
            # Calculate correlation dimension
            features = self._calculate_correlation_dimension_features(returns, features, nonlinear_window)
            
            # Calculate recurrence quantification analysis
            features = self._calculate_rqa_features(returns, features, nonlinear_window)
            
            # Calculate chaos detection measures
            features = self._calculate_chaos_detection(returns, features, nonlinear_window)
            
            # Calculate attractor analysis
            features = self._calculate_attractor_features(returns, features, nonlinear_window)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating nonlinear dynamics features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_lyapunov_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Lyapunov exponent features using multiple methods."""
        
        # Largest Lyapunov exponent (simplified method)
        def lyapunov_largest_func(x):
            return self._estimate_lyapunov_largest(x)
        
        features['lyapunov_largest'] = self.rolling_apply_safe(
            returns, window, lyapunov_largest_func
        )
        
        # Wolf method
        def lyapunov_wolf_func(x):
            return self._lyapunov_wolf_method(x)
        
        features['lyapunov_wolf'] = self.rolling_apply_safe(
            returns, window, lyapunov_wolf_func
        )
        
        # Rosenstein method
        def lyapunov_rosenstein_func(x):
            return self._lyapunov_rosenstein_method(x)
        
        features['lyapunov_rosenstein'] = self.rolling_apply_safe(
            returns, window, lyapunov_rosenstein_func
        )
        
        # Kantz method
        def lyapunov_kantz_func(x):
            return self._lyapunov_kantz_method(x)
        
        features['lyapunov_kantz'] = self.rolling_apply_safe(
            returns, window, lyapunov_kantz_func
        )
        
        return features
    
    def _calculate_correlation_dimension_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate correlation dimension features."""
        def correlation_dim_func(x):
            return self._calculate_correlation_dimension(x)
        
        features['correlation_dimension'] = self.rolling_apply_safe(
            returns, window, correlation_dim_func
        )
        
        return features
    
    def _calculate_rqa_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Recurrence Quantification Analysis features."""
        def rqa_func(x):
            return self._calculate_rqa_measures(x)
        
        # Calculate all RQA measures at once for efficiency
        rqa_results = self.rolling_apply_safe(returns, window, rqa_func)
        
        # Extract individual RQA measures (returned as tuple)
        def extract_rqa_measure(results, index):
            if isinstance(results, (list, tuple)) and len(results) > index:
                return results[index]
            return np.nan
        
        # Apply extraction for each measure
        features['recurrence_rate'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 0) if not pd.isna(x) else np.nan)
        features['determinism'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 1) if not pd.isna(x) else np.nan)
        features['laminarity'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 2) if not pd.isna(x) else np.nan)
        features['entropy_rqa'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 3) if not pd.isna(x) else np.nan)
        features['max_line_length'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 4) if not pd.isna(x) else np.nan)
        features['divergence'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 5) if not pd.isna(x) else np.nan)
        features['trend_rqa'] = rqa_results.apply(lambda x: extract_rqa_measure(x, 6) if not pd.isna(x) else np.nan)
        
        return features
    
    def _calculate_chaos_detection(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate chaos detection measures."""
        # 0-1 test for chaos
        def chaos_01_func(x):
            return self._chaos_01_test(x)
        
        features['chaos_test_01'] = self.rolling_apply_safe(
            returns, window, chaos_01_func
        )
        
        # Optimal embedding dimension
        def embedding_dim_func(x):
            return self._estimate_embedding_dimension(x)
        
        features['embedding_dimension_opt'] = self.rolling_apply_safe(
            returns, window, embedding_dim_func
        )
        
        # False nearest neighbors
        def fnn_func(x):
            return self._false_nearest_neighbors(x)
        
        features['false_nearest_neighbors'] = self.rolling_apply_safe(
            returns, window, fnn_func
        )
        
        return features
    
    def _calculate_attractor_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate attractor analysis features."""
        # Attractor dimension
        def attractor_dim_func(x):
            return self._estimate_attractor_dimension(x)
        
        features['attractor_dimension'] = self.rolling_apply_safe(
            returns, window, attractor_dim_func
        )
        
        # Predictability measure
        def predictability_func(x):
            return self._calculate_predictability(x)
        
        features['predictability'] = self.rolling_apply_safe(
            returns, window, predictability_func
        )
        
        # LMC complexity
        def lmc_complexity_func(x):
            return self._lmc_complexity(x)
        
        features['complexity_lmc'] = self.rolling_apply_safe(
            returns, window, lmc_complexity_func
        )
        
        return features
    
    def _embed_time_series(self, data: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
        """Create time-delay embedded vectors."""
        if len(data) < m * tau:
            return np.array([]).reshape(0, m)
        
        n = len(data) - (m - 1) * tau
        embedded = np.zeros((n, m))
        
        for i in range(m):
            embedded[:, i] = data[i * tau:i * tau + n]
        
        return embedded
    
    def _estimate_lyapunov_largest(self, data: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent using simplified method."""
        if len(data) < 50:
            return np.nan
        
        try:
            # Phase space reconstruction
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 20:
                return np.nan
            
            # Find nearest neighbors and track divergence
            divergences = []
            
            for i in range(len(embedded) - self.lyapunov_k):
                # Find nearest neighbor
                distances = np.linalg.norm(embedded[i] - embedded[i+1:], axis=1)
                
                if len(distances) == 0:
                    continue
                
                min_idx = np.argmin(distances) + i + 1
                initial_dist = distances[min_idx - i - 1]
                
                # Track evolution
                if (min_idx + self.lyapunov_k < len(embedded) and 
                    i + self.lyapunov_k < len(embedded) and 
                    initial_dist > self.numerical_tolerance):
                    
                    final_dist = np.linalg.norm(embedded[i + self.lyapunov_k] - embedded[min_idx + self.lyapunov_k])
                    
                    if final_dist > self.numerical_tolerance:
                        divergence = np.log(final_dist / initial_dist) / self.lyapunov_k
                        if np.isfinite(divergence):
                            divergences.append(divergence)
            
            return np.mean(divergences) if divergences else np.nan
            
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _lyapunov_wolf_method(self, data: np.ndarray) -> float:
        """Calculate Lyapunov exponent using Wolf method."""
        if len(data) < 50:
            return np.nan
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 30:
                return np.nan
            
            # Wolf algorithm
            sum_log_divergence = 0
            count = 0
            
            for i in range(len(embedded) - self.lyapunov_k - 1):
                # Find nearest neighbor
                current_point = embedded[i]
                distances = np.linalg.norm(embedded[i+1:] - current_point, axis=1)
                
                if len(distances) == 0:
                    continue
                
                min_idx = np.argmin(distances) + i + 1
                initial_distance = distances[min_idx - i - 1]
                
                if initial_distance < self.numerical_tolerance:
                    continue
                
                # Evolve for k steps
                if min_idx + self.lyapunov_k < len(embedded):
                    evolved_distance = np.linalg.norm(
                        embedded[i + self.lyapunov_k] - embedded[min_idx + self.lyapunov_k]
                    )
                    
                    if evolved_distance > self.numerical_tolerance:
                        sum_log_divergence += np.log(evolved_distance / initial_distance)
                        count += 1
            
            return sum_log_divergence / (count * self.lyapunov_k) if count > 0 else np.nan
            
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _lyapunov_rosenstein_method(self, data: np.ndarray) -> float:
        """Calculate Lyapunov exponent using Rosenstein method."""
        if len(data) < 50:
            return np.nan
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 30:
                return np.nan
            
            # Calculate average logarithmic divergence
            max_evolution = min(self.lyapunov_k, len(embedded) // 10)
            divergence_sums = np.zeros(max_evolution)
            counts = np.zeros(max_evolution)
            
            for i in range(len(embedded) - max_evolution):
                # Find nearest neighbor
                distances = np.linalg.norm(embedded[i] - embedded, axis=1)
                distances[i] = np.inf  # Exclude self
                
                nearest_idx = np.argmin(distances)
                
                # Track divergence evolution
                for j in range(1, max_evolution):
                    if i + j < len(embedded) and nearest_idx + j < len(embedded):
                        dist = np.linalg.norm(embedded[i + j] - embedded[nearest_idx + j])
                        
                        if dist > self.numerical_tolerance:
                            divergence_sums[j] += np.log(dist)
                            counts[j] += 1
            
            # Calculate Lyapunov exponent from slope
            valid_points = counts > 0
            if np.sum(valid_points) >= 3:
                avg_divergence = divergence_sums[valid_points] / counts[valid_points]
                time_steps = np.arange(1, max_evolution)[valid_points]
                
                # Linear fit
                slope = np.polyfit(time_steps, avg_divergence, 1)[0]
                return slope
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _lyapunov_kantz_method(self, data: np.ndarray) -> float:
        """Calculate Lyapunov exponent using Kantz method."""
        if len(data) < 50:
            return np.nan
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 30:
                return np.nan
            
            # Kantz algorithm with neighborhood constraint
            max_evolution = min(self.lyapunov_k, len(embedded) // 10)
            S_values = []
            
            for evolution_time in range(1, max_evolution + 1):
                sum_log_divergence = 0
                count = 0
                
                for i in range(len(embedded) - evolution_time):
                    # Find neighbors within epsilon
                    epsilon = np.std(data) * 0.1  # Adaptive epsilon
                    distances = np.linalg.norm(embedded[i] - embedded, axis=1)
                    
                    neighbors = np.where((distances < epsilon) & (distances > 0))[0]
                    
                    if len(neighbors) >= self.lyapunov_min_neighbors:
                        divergence_sum = 0
                        valid_neighbors = 0
                        
                        for neighbor_idx in neighbors:
                            if neighbor_idx + evolution_time < len(embedded):
                                final_distance = np.linalg.norm(
                                    embedded[i + evolution_time] - embedded[neighbor_idx + evolution_time]
                                )
                                
                                if final_distance > self.numerical_tolerance:
                                    divergence_sum += final_distance
                                    valid_neighbors += 1
                        
                        if valid_neighbors > 0:
                            avg_divergence = divergence_sum / valid_neighbors
                            sum_log_divergence += np.log(avg_divergence)
                            count += 1
                
                if count > 0:
                    S_values.append(sum_log_divergence / count)
            
            # Calculate slope
            if len(S_values) >= 3:
                time_steps = np.arange(1, len(S_values) + 1)
                slope = np.polyfit(time_steps, S_values, 1)[0]
                return slope
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_correlation_dimension(self, data: np.ndarray) -> float:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm."""
        if len(data) < self.embedding_dimension * 10:
            return np.nan
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 20:
                return np.nan
            
            # Calculate correlation sums for different radii
            data_std = np.std(data)
            radii = np.logspace(
                np.log10(self.correlation_r_min * data_std),
                np.log10(self.correlation_r_max * data_std),
                self.correlation_n_radii
            )
            
            correlation_sums = []
            n_points = len(embedded)
            
            for r in radii:
                count = 0
                for i in range(n_points):
                    distances = np.linalg.norm(embedded[i] - embedded[i+1:], axis=1)
                    count += np.sum(distances < r)
                
                # Normalize
                total_pairs = n_points * (n_points - 1) / 2
                correlation_sums.append(count / total_pairs if total_pairs > 0 else 0)
            
            # Find scaling region and calculate dimension
            log_r = np.log(radii)
            log_c = np.log(np.array(correlation_sums) + 1e-10)
            
            # Identify linear scaling region
            valid_indices = (log_c > -10) & (log_c < -1) & np.isfinite(log_c)
            
            if np.sum(valid_indices) >= 3:
                slope = np.polyfit(log_r[valid_indices], log_c[valid_indices], 1)[0]
                return slope
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_rqa_measures(self, data: np.ndarray) -> Tuple[float, ...]:
        """Calculate Recurrence Quantification Analysis measures."""
        if len(data) < 30:
            return tuple([np.nan] * 7)
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 20:
                return tuple([np.nan] * 7)
            
            # Create recurrence matrix
            n = len(embedded)
            recurrence_matrix = np.zeros((n, n))
            
            # Set threshold for recurrence
            threshold = self.recurrence_threshold * np.std(data)
            
            for i in range(n):
                distances = np.linalg.norm(embedded[i] - embedded, axis=1)
                recurrence_matrix[i, :] = (distances < threshold).astype(int)
            
            # Calculate RQA measures
            rr = self._calculate_recurrence_rate(recurrence_matrix)
            det = self._calculate_determinism(recurrence_matrix)
            lam = self._calculate_laminarity(recurrence_matrix)
            entropy = self._calculate_entropy_rqa(recurrence_matrix)
            max_line = self._calculate_max_line_length(recurrence_matrix)
            div = self._calculate_divergence(recurrence_matrix)
            trend = self._calculate_trend_rqa(recurrence_matrix)
            
            return (rr, det, lam, entropy, max_line, div, trend)
            
        except (ValueError, RuntimeWarning):
            return tuple([np.nan] * 7)
    
    def _calculate_recurrence_rate(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate recurrence rate."""
        n = recurrence_matrix.shape[0]
        total_points = n * n
        recurrent_points = np.sum(recurrence_matrix)
        return recurrent_points / total_points if total_points > 0 else np.nan
    
    def _calculate_determinism(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate determinism from recurrence matrix."""
        n = recurrence_matrix.shape[0]
        
        # Find diagonal lines
        diagonal_lengths = []
        
        for offset in range(-(n-1), n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            
            # Find line segments
            line_length = 0
            for point in diagonal:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= self.recurrence_min_line_length:
                        diagonal_lengths.append(line_length)
                    line_length = 0
            
            # Check final segment
            if line_length >= self.recurrence_min_line_length:
                diagonal_lengths.append(line_length)
        
        total_recurrent_points = np.sum(recurrence_matrix)
        points_in_lines = sum(diagonal_lengths)
        
        return points_in_lines / total_recurrent_points if total_recurrent_points > 0 else np.nan
    
    def _calculate_laminarity(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate laminarity from recurrence matrix."""
        n = recurrence_matrix.shape[0]
        
        # Find vertical lines
        vertical_lengths = []
        
        for col in range(n):
            column = recurrence_matrix[:, col]
            
            # Find line segments
            line_length = 0
            for point in column:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= self.recurrence_min_line_length:
                        vertical_lengths.append(line_length)
                    line_length = 0
            
            # Check final segment
            if line_length >= self.recurrence_min_line_length:
                vertical_lengths.append(line_length)
        
        total_recurrent_points = np.sum(recurrence_matrix)
        points_in_vertical_lines = sum(vertical_lengths)
        
        return points_in_vertical_lines / total_recurrent_points if total_recurrent_points > 0 else np.nan
    
    def _calculate_entropy_rqa(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate entropy from diagonal line distribution."""
        n = recurrence_matrix.shape[0]
        diagonal_lengths = []
        
        for offset in range(-(n-1), n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            
            line_length = 0
            for point in diagonal:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= self.recurrence_min_line_length:
                        diagonal_lengths.append(line_length)
                    line_length = 0
            
            if line_length >= self.recurrence_min_line_length:
                diagonal_lengths.append(line_length)
        
        if not diagonal_lengths:
            return np.nan
        
        # Calculate entropy of line length distribution
        length_counts = {}
        for length in diagonal_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        total_lines = len(diagonal_lengths)
        entropy = 0
        
        for count in length_counts.values():
            prob = count / total_lines
            entropy -= prob * np.log2(prob + 1e-10)
        
        return entropy
    
    def _calculate_max_line_length(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate maximum diagonal line length."""
        n = recurrence_matrix.shape[0]
        max_length = 0
        
        for offset in range(-(n-1), n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            
            current_length = 0
            for point in diagonal:
                if point == 1:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0
        
        return float(max_length)
    
    def _calculate_divergence(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate divergence (inverse of max diagonal line length)."""
        max_line = self._calculate_max_line_length(recurrence_matrix)
        return 1.0 / max_line if max_line > 0 else np.nan
    
    def _calculate_trend_rqa(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate trend in recurrence matrix."""
        n = recurrence_matrix.shape[0]
        
        # Calculate recurrence rate for each diagonal
        diagonal_rr = []
        diagonal_indices = []
        
        for offset in range(-(n//2), n//2):
            diagonal = np.diag(recurrence_matrix, k=offset)
            if len(diagonal) > 10:  # Sufficient length
                rr = np.mean(diagonal)
                diagonal_rr.append(rr)
                diagonal_indices.append(offset)
        
        if len(diagonal_rr) < 3:
            return np.nan
        
        # Calculate trend (slope)
        try:
            slope = np.polyfit(diagonal_indices, diagonal_rr, 1)[0]
            return slope
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _chaos_01_test(self, data: np.ndarray) -> float:
        """Perform 0-1 test for chaos."""
        if len(data) < 50:
            return np.nan
        
        try:
            n = len(data)
            c = np.pi  # Arbitrary constant
            
            # Calculate translation variables
            p = np.zeros(n)
            q = np.zeros(n)
            
            for i in range(1, n):
                p[i] = p[i-1] + data[i] * np.cos(i * c)
                q[i] = q[i-1] + data[i] * np.sin(i * c)
            
            # Calculate mean square displacement
            M = np.zeros(n//10)  # Use subset for efficiency
            
            for n_steps in range(1, len(M) + 1):
                displacement_sum = 0
                count = 0
                
                for j in range(n - n_steps):
                    displacement = (p[j + n_steps] - p[j])**2 + (q[j + n_steps] - q[j])**2
                    displacement_sum += displacement
                    count += 1
                
                M[n_steps - 1] = displacement_sum / count if count > 0 else 0
            
            # Calculate asymptotic growth rate
            if len(M) >= 3:
                # Linear fit in log-log space
                time_steps = np.arange(1, len(M) + 1)
                valid_indices = M > 0
                
                if np.sum(valid_indices) >= 3:
                    log_time = np.log(time_steps[valid_indices])
                    log_M = np.log(M[valid_indices])
                    
                    slope = np.polyfit(log_time, log_M, 1)[0]
                    
                    # K = 0 for regular motion, K = 1 for chaotic motion
                    K = slope / 2.0
                    return K
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _estimate_embedding_dimension(self, data: np.ndarray) -> float:
        """Estimate optimal embedding dimension using false nearest neighbors."""
        if len(data) < 30:
            return np.nan
        
        try:
            max_dim = min(self.max_embedding_dim, len(data) // 10)
            fnn_percentages = []
            
            for m in range(1, max_dim + 1):
                fnn_pct = self._false_nearest_neighbors_fraction(data, m)
                fnn_percentages.append(fnn_pct)
                
                # Stop if FNN percentage is sufficiently low
                if fnn_pct < 0.01:  # 1% threshold
                    return float(m)
            
            # Find minimum or use criterion
            if fnn_percentages:
                min_idx = np.argmin(fnn_percentages)
                return float(min_idx + 1)
            
            return float(self.embedding_dimension)
            
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _false_nearest_neighbors(self, data: np.ndarray) -> float:
        """Calculate false nearest neighbors percentage."""
        return self._false_nearest_neighbors_fraction(data, self.embedding_dimension)
    
    def _false_nearest_neighbors_fraction(self, data: np.ndarray, m: int) -> float:
        """Calculate fraction of false nearest neighbors for given embedding dimension."""
        if len(data) < (m + 1) * self.time_delay + 10:
            return np.nan
        
        try:
            # Embed in m and m+1 dimensions
            embedded_m = self._embed_time_series(data, m, self.time_delay)
            embedded_m1 = self._embed_time_series(data, m + 1, self.time_delay)
            
            if len(embedded_m) < 10 or len(embedded_m1) < 10:
                return np.nan
            
            # Align lengths
            min_length = min(len(embedded_m), len(embedded_m1))
            embedded_m = embedded_m[:min_length]
            embedded_m1 = embedded_m1[:min_length]
            
            false_neighbors = 0
            total_neighbors = 0
            
            tolerance = 15.0  # Standard tolerance
            
            for i in range(min_length):
                # Find nearest neighbor in m-dimensional space
                distances_m = np.linalg.norm(embedded_m[i] - embedded_m, axis=1)
                distances_m[i] = np.inf  # Exclude self
                
                nearest_idx = np.argmin(distances_m)
                nearest_distance_m = distances_m[nearest_idx]
                
                if nearest_distance_m > 0:
                    # Check distance in (m+1)-dimensional space
                    distance_m1 = np.linalg.norm(embedded_m1[i] - embedded_m1[nearest_idx])
                    
                    # False neighbor criterion
                    ratio = distance_m1 / nearest_distance_m
                    
                    if ratio > tolerance:
                        false_neighbors += 1
                    
                    total_neighbors += 1
            
            return false_neighbors / total_neighbors if total_neighbors > 0 else np.nan
            
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _estimate_attractor_dimension(self, data: np.ndarray) -> float:
        """Estimate attractor dimension using box counting."""
        if len(data) < 50:
            return np.nan
        
        try:
            # Embed time series
            embedded = self._embed_time_series(data, self.embedding_dimension, self.time_delay)
            
            if len(embedded) < 20:
                return np.nan
            
            # Normalize embedded vectors
            embedded_norm = (embedded - np.mean(embedded, axis=0)) / np.std(embedded, axis=0)
            
            # Box counting with different box sizes
            box_sizes = np.logspace(-1, 0, 10)  # From 0.1 to 1.0
            box_counts = []
            
            for box_size in box_sizes:
                # Discretize space
                n_boxes_per_dim = int(1.0 / box_size)
                if n_boxes_per_dim < 2:
                    continue
                
                # Count occupied boxes
                occupied_boxes = set()
                
                for point in embedded_norm:
                    box_indices = tuple(np.floor(point * n_boxes_per_dim).astype(int))
                    occupied_boxes.add(box_indices)
                
                box_counts.append(len(occupied_boxes))
            
            # Calculate dimension from scaling
            if len(box_counts) >= 3:
                valid_sizes = box_sizes[:len(box_counts)]
                log_sizes = np.log(1.0 / valid_sizes)
                log_counts = np.log(box_counts)
                
                # Linear fit
                dimension = np.polyfit(log_sizes, log_counts, 1)[0]
                return dimension
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_predictability(self, data: np.ndarray) -> float:
        """Calculate predictability measure."""
        if len(data) < 30:
            return np.nan
        
        try:
            # Use simple AR model predictability
            n_train = len(data) // 2
            train_data = data[:n_train]
            test_data = data[n_train:]
            
            # Fit AR(1) model
            if len(train_data) < 3:
                return np.nan
            
            X = train_data[:-1]
            y = train_data[1:]
            
            # Simple linear regression
            if np.var(X) < self.numerical_tolerance:
                return np.nan
            
            slope = np.cov(X, y)[0, 1] / np.var(X)
            intercept = np.mean(y) - slope * np.mean(X)
            
            # Predict test data
            predictions = []
            for i in range(len(test_data) - 1):
                if i == 0:
                    pred = slope * train_data[-1] + intercept
                else:
                    pred = slope * test_data[i-1] + intercept
                predictions.append(pred)
            
            if len(predictions) > 0:
                # Calculate prediction accuracy
                actual = test_data[1:len(predictions)+1]
                mse = np.mean((np.array(predictions) - actual)**2)
                variance = np.var(actual)
                
                # Predictability as 1 - normalized MSE
                predictability = 1 - mse / (variance + self.numerical_tolerance)
                return max(0, min(1, predictability))
            
            return np.nan
            
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _lmc_complexity(self, data: np.ndarray) -> float:
        """Calculate LMC (López-Mancini-Calbet) complexity."""
        if len(data) < 20:
            return np.nan
        
        try:
            # Normalize data
            data_norm = (data - np.mean(data)) / (np.std(data) + self.numerical_tolerance)
            
            # Calculate probability distribution
            hist, _ = np.histogram(data_norm, bins=int(np.sqrt(len(data_norm))))
            prob = hist / np.sum(hist)
            prob = prob[prob > 0]  # Remove zero probabilities
            
            if len(prob) <= 1:
                return np.nan
            
            # Calculate normalized Shannon entropy
            H = -np.sum(prob * np.log2(prob)) / np.log2(len(prob))
            
            # Calculate disequilibrium
            uniform_prob = 1.0 / len(prob)
            D = np.sum((prob - uniform_prob)**2)
            
            # LMC complexity
            complexity = H * D
            
            return complexity
            
        except (ValueError, RuntimeWarning):
            return np.nan