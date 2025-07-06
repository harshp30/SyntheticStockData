"""
Validation Module
Compare synthetic data with historical data using statistical tests and metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import jarque_bera
import warnings

# Configuration is now handled directly in the code
from data_loader import HistoricalDataLoader

class SyntheticDataValidator:
    """Validates synthetic data against historical benchmarks"""
    
    def __init__(self, historical_ticker: str = "^GSPC"):
        self.historical_ticker = historical_ticker
        self.historical_data = None
        self.historical_stats = None
        self.validation_results = {}
        
    def load_historical_data(self):
        """Load historical data for comparison"""
        print(f"Loading historical data for {self.historical_ticker}...")
        
        loader = HistoricalDataLoader(self.historical_ticker)
        loader.fetch_data()
        loader.identify_cycles()
        loader.calculate_statistics()
        
        self.historical_data = loader.data
        self.historical_stats = loader.statistics
        
        print("Historical data loaded successfully")
        
    def validate_synthetic_data(self, synthetic_data: pd.DataFrame) -> Dict:
        """Comprehensive validation of synthetic data against stylized facts"""
        
        if self.historical_data is None:
            self.load_historical_data()
            
        print("Validating synthetic data...")
        
        # Initialize results with enhanced validation
        validation_results = {
            'return_distribution': self._validate_return_distribution(synthetic_data),
            'volatility_structure': self._validate_volatility_structure(synthetic_data),
            'regime_characteristics': self._validate_regime_characteristics(synthetic_data),
            'risk_metrics': self._validate_risk_metrics(synthetic_data),
            'autocorrelation': self._validate_autocorrelation(synthetic_data),
            'distributional_properties': self._validate_distributional_properties(synthetic_data),
            'leverage_effect': self._validate_leverage_effect(synthetic_data),
            'fat_tails': self._validate_fat_tails(synthetic_data),
            'volatility_clustering': self._validate_volatility_clustering_detailed(synthetic_data),
            'stylized_facts': self._validate_stylized_facts(synthetic_data),
            'summary_statistics': self._validate_summary_statistics(synthetic_data)
        }
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        
        self.validation_results = validation_results
        
        print("Validation completed")
        return validation_results
    
    def _validate_return_distribution(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate return distribution properties"""
        
        # Get returns
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Basic moments
        hist_mean = hist_returns.mean()
        hist_std = hist_returns.std()
        hist_skew = hist_returns.skew()
        hist_kurt = hist_returns.kurtosis()
        
        synth_mean = synth_returns.mean()
        synth_std = synth_returns.std()
        synth_skew = synth_returns.skew()
        synth_kurt = synth_returns.kurtosis()
        
        # Statistical tests
        ks_stat, ks_pvalue = stats.ks_2samp(hist_returns, synth_returns)
        
        # Kolmogorov-Smirnov test for normality
        hist_norm_stat, hist_norm_pvalue = stats.jarque_bera(hist_returns)
        synth_norm_stat, synth_norm_pvalue = stats.jarque_bera(synth_returns)
        
        return {
            'moments': {
                'mean_diff': abs(synth_mean - hist_mean),
                'std_diff': abs(synth_std - hist_std),
                'skew_diff': abs(synth_skew - hist_skew),
                'kurt_diff': abs(synth_kurt - hist_kurt),
                'mean_ratio': synth_mean / hist_mean if hist_mean != 0 else np.nan,
                'std_ratio': synth_std / hist_std if hist_std != 0 else np.nan
            },
            'statistical_tests': {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'distributions_similar': ks_pvalue > VALIDATION_CONFIG.significance_level
            },
            'normality_tests': {
                'historical_normal': hist_norm_pvalue > VALIDATION_CONFIG.significance_level,
                'synthetic_normal': synth_norm_pvalue > VALIDATION_CONFIG.significance_level,
                'both_non_normal': (hist_norm_pvalue <= VALIDATION_CONFIG.significance_level and 
                                   synth_norm_pvalue <= VALIDATION_CONFIG.significance_level)
            }
        }
    
    def _validate_volatility_structure(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate volatility clustering and structure"""
        
        # Calculate rolling volatilities
        hist_vol_21 = self.historical_data['Returns'].rolling(21).std() * np.sqrt(252)
        synth_vol_21 = synthetic_data['simple_return'].rolling(21).std() * np.sqrt(252)
        
        hist_vol_252 = self.historical_data['Returns'].rolling(252).std() * np.sqrt(252)
        synth_vol_252 = synthetic_data['simple_return'].rolling(252).std() * np.sqrt(252)
        
        # Compare volatility distributions
        hist_vol_21_clean = hist_vol_21.dropna()
        synth_vol_21_clean = synth_vol_21.dropna()
        
        # Volatility clustering test (autocorrelation of squared returns)
        hist_sq_returns = self.historical_data['Returns'].dropna() ** 2
        synth_sq_returns = synthetic_data['simple_return'].dropna() ** 2
        
        hist_vol_cluster = self._calculate_autocorrelation(hist_sq_returns, max_lags=20)
        synth_vol_cluster = self._calculate_autocorrelation(synth_sq_returns, max_lags=20)
        
        return {
            'volatility_levels': {
                'historical_mean_vol': hist_vol_21_clean.mean(),
                'synthetic_mean_vol': synth_vol_21_clean.mean(),
                'vol_difference': abs(synth_vol_21_clean.mean() - hist_vol_21_clean.mean()),
                'vol_ratio': synth_vol_21_clean.mean() / hist_vol_21_clean.mean() if hist_vol_21_clean.mean() != 0 else np.nan
            },
            'volatility_clustering': {
                'historical_autocorr_mean': np.mean(hist_vol_cluster),
                'synthetic_autocorr_mean': np.mean(synth_vol_cluster),
                'clustering_similarity': 1 - abs(np.mean(hist_vol_cluster) - np.mean(synth_vol_cluster))
            },
            'volatility_distribution': {
                'hist_vol_std': hist_vol_21_clean.std(),
                'synth_vol_std': synth_vol_21_clean.std(),
                'vol_variability_ratio': synth_vol_21_clean.std() / hist_vol_21_clean.std() if hist_vol_21_clean.std() != 0 else np.nan
            }
        }
    
    def _validate_regime_characteristics(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate regime switching characteristics"""
        
        # This is a simplified validation - would need regime identification for historical data
        hist_stats = self.historical_stats
        
        # Calculate synthetic regime statistics
        synth_bull_data = synthetic_data[synthetic_data['regime'] == 'bull']
        synth_bear_data = synthetic_data[synthetic_data['regime'] == 'bear']
        
        synth_bull_return = synth_bull_data['simple_return'].mean() * 252 if len(synth_bull_data) > 0 else 0
        synth_bear_return = synth_bear_data['simple_return'].mean() * 252 if len(synth_bear_data) > 0 else 0
        
        synth_bull_vol = synth_bull_data['simple_return'].std() * np.sqrt(252) if len(synth_bull_data) > 0 else 0
        synth_bear_vol = synth_bear_data['simple_return'].std() * np.sqrt(252) if len(synth_bear_data) > 0 else 0
        
        # Compare with historical regime stats if available
        regime_comparison = {}
        if hist_stats and 'bull_markets' in hist_stats and 'bear_markets' in hist_stats:
            hist_bull_stats = hist_stats['bull_markets']
            hist_bear_stats = hist_stats['bear_markets']
            
            if hist_bull_stats and hist_bear_stats:
                regime_comparison = {
                    'bull_return_diff': abs(synth_bull_return - hist_bull_stats.get('avg_annual_return', 0)),
                    'bear_return_diff': abs(synth_bear_return - hist_bear_stats.get('avg_annual_return', 0)),
                    'bull_vol_diff': abs(synth_bull_vol - hist_bull_stats.get('volatility', 0)),
                    'bear_vol_diff': abs(synth_bear_vol - hist_bear_stats.get('volatility', 0))
                }
        
        return {
            'synthetic_regime_stats': {
                'bull_annual_return': synth_bull_return,
                'bear_annual_return': synth_bear_return,
                'bull_volatility': synth_bull_vol,
                'bear_volatility': synth_bear_vol,
                'regime_volatility_ratio': synth_bear_vol / synth_bull_vol if synth_bull_vol != 0 else np.nan
            },
            'historical_comparison': regime_comparison
        }
    
    def _validate_risk_metrics(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate risk metrics"""
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Value at Risk
        hist_var_95 = hist_returns.quantile(0.05)
        hist_var_99 = hist_returns.quantile(0.01)
        synth_var_95 = synth_returns.quantile(0.05)
        synth_var_99 = synth_returns.quantile(0.01)
        
        # Maximum drawdown
        hist_dd = self.historical_data['Drawdown'].min()
        synth_dd = synthetic_data['drawdown'].min()
        
        # Sharpe ratio
        hist_sharpe = (hist_returns.mean() * 252) / (hist_returns.std() * np.sqrt(252))
        synth_sharpe = (synth_returns.mean() * 252) / (synth_returns.std() * np.sqrt(252))
        
        return {
            'var_metrics': {
                'hist_var_95': hist_var_95,
                'synth_var_95': synth_var_95,
                'var_95_diff': abs(synth_var_95 - hist_var_95),
                'hist_var_99': hist_var_99,
                'synth_var_99': synth_var_99,
                'var_99_diff': abs(synth_var_99 - hist_var_99)
            },
            'drawdown_metrics': {
                'hist_max_dd': hist_dd,
                'synth_max_dd': synth_dd,
                'dd_diff': abs(synth_dd - hist_dd)
            },
            'performance_metrics': {
                'hist_sharpe': hist_sharpe,
                'synth_sharpe': synth_sharpe,
                'sharpe_diff': abs(synth_sharpe - hist_sharpe)
            }
        }
    
    def _validate_autocorrelation(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate autocorrelation structure"""
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Calculate autocorrelations
        hist_autocorr = self._calculate_autocorrelation(hist_returns, max_lags=20)
        synth_autocorr = self._calculate_autocorrelation(synth_returns, max_lags=20)
        
        # Compare autocorrelation structures
        autocorr_mse = mean_squared_error(hist_autocorr, synth_autocorr)
        autocorr_mae = mean_absolute_error(hist_autocorr, synth_autocorr)
        
        return {
            'autocorrelation_comparison': {
                'hist_autocorr_mean': np.mean(hist_autocorr),
                'synth_autocorr_mean': np.mean(synth_autocorr),
                'autocorr_mse': autocorr_mse,
                'autocorr_mae': autocorr_mae,
                'autocorr_similarity': 1 - autocorr_mse  # Simple similarity metric
            }
        }
    
    def _validate_distributional_properties(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate distributional properties"""
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Quantile comparison
        percentiles = VALIDATION_CONFIG.return_percentiles
        hist_quantiles = [hist_returns.quantile(p/100) for p in percentiles]
        synth_quantiles = [synth_returns.quantile(p/100) for p in percentiles]
        
        quantile_differences = [abs(h - s) for h, s in zip(hist_quantiles, synth_quantiles)]
        
        # Tail behavior
        hist_extreme_neg = (hist_returns < hist_returns.quantile(0.05)).sum()
        synth_extreme_neg = (synth_returns < synth_returns.quantile(0.05)).sum()
        
        hist_extreme_pos = (hist_returns > hist_returns.quantile(0.95)).sum()
        synth_extreme_pos = (synth_returns > synth_returns.quantile(0.95)).sum()
        
        return {
            'quantile_comparison': {
                'percentiles': percentiles,
                'historical_quantiles': hist_quantiles,
                'synthetic_quantiles': synth_quantiles,
                'quantile_differences': quantile_differences,
                'mean_quantile_diff': np.mean(quantile_differences)
            },
            'tail_behavior': {
                'hist_extreme_negative_count': hist_extreme_neg,
                'synth_extreme_negative_count': synth_extreme_neg,
                'hist_extreme_positive_count': hist_extreme_pos,
                'synth_extreme_positive_count': synth_extreme_pos,
                'extreme_event_ratio': (synth_extreme_neg + synth_extreme_pos) / (hist_extreme_neg + hist_extreme_pos) if (hist_extreme_neg + hist_extreme_pos) > 0 else np.nan
            }
        }
    
    def _validate_summary_statistics(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate summary statistics"""
        
        hist_stats = self.historical_stats['overall']
        
        synth_returns = synthetic_data['simple_return'].dropna()
        synth_annual_return = synth_returns.mean() * 252
        synth_annual_vol = synth_returns.std() * np.sqrt(252)
        synth_sharpe = synth_annual_return / synth_annual_vol if synth_annual_vol != 0 else np.nan
        synth_skew = synth_returns.skew()
        synth_kurt = synth_returns.kurtosis()
        synth_max_dd = synthetic_data['drawdown'].min()
        
        return {
            'performance_comparison': {
                'hist_annual_return': hist_stats['annual_return'],
                'synth_annual_return': synth_annual_return,
                'return_difference': abs(synth_annual_return - hist_stats['annual_return']),
                'hist_annual_volatility': hist_stats['annual_volatility'],
                'synth_annual_volatility': synth_annual_vol,
                'volatility_difference': abs(synth_annual_vol - hist_stats['annual_volatility']),
                'hist_sharpe': hist_stats['sharpe_ratio'],
                'synth_sharpe': synth_sharpe,
                'sharpe_difference': abs(synth_sharpe - hist_stats['sharpe_ratio'])
            },
            'distribution_comparison': {
                'hist_skewness': hist_stats['skewness'],
                'synth_skewness': synth_skew,
                'skewness_difference': abs(synth_skew - hist_stats['skewness']),
                'hist_kurtosis': hist_stats['kurtosis'],
                'synth_kurtosis': synth_kurt,
                'kurtosis_difference': abs(synth_kurt - hist_stats['kurtosis'])
            },
            'risk_comparison': {
                'hist_max_drawdown': hist_stats['max_drawdown'],
                'synth_max_drawdown': synth_max_dd,
                'drawdown_difference': abs(synth_max_dd - hist_stats['max_drawdown'])
            }
        }
    
    def _calculate_autocorrelation(self, series: pd.Series, max_lags: int = 20) -> List[float]:
        """Calculate autocorrelation for different lags"""
        
        autocorr = []
        for lag in range(1, max_lags + 1):
            corr = series.autocorr(lag=lag)
            autocorr.append(corr if not np.isnan(corr) else 0)
            
        return autocorr
    
    def _calculate_overall_score(self, results: Dict) -> Dict:
        """Calculate overall validation score"""
        
        scores = {}
        
        # Return distribution score
        return_dist = results['return_distribution']
        if return_dist['statistical_tests']['distributions_similar']:
            scores['return_distribution'] = 1.0
        else:
            # Penalize based on moment differences
            moment_score = 1.0 - min(1.0, (
                return_dist['moments']['mean_diff'] * 10 +
                return_dist['moments']['std_diff'] * 10 +
                abs(return_dist['moments']['skew_diff']) * 0.5 +
                abs(return_dist['moments']['kurt_diff']) * 0.1
            ))
            scores['return_distribution'] = max(0.0, moment_score)
        
        # Volatility structure score
        vol_struct = results['volatility_structure']
        vol_score = 1.0 - min(1.0, vol_struct['volatility_levels']['vol_difference'] * 10)
        scores['volatility_structure'] = max(0.0, vol_score)
        
        # Risk metrics score
        risk_metrics = results['risk_metrics']
        risk_score = 1.0 - min(1.0, (
            risk_metrics['var_metrics']['var_95_diff'] * 20 +
            risk_metrics['var_metrics']['var_99_diff'] * 20 +
            abs(risk_metrics['drawdown_metrics']['dd_diff']) * 2
        ))
        scores['risk_metrics'] = max(0.0, risk_score)
        
        # Overall score (weighted average)
        weights = {
            'return_distribution': 0.3,
            'volatility_structure': 0.25,
            'risk_metrics': 0.25,
            'regime_characteristics': 0.1,
            'autocorrelation': 0.1
        }
        
        # Add missing scores with default values
        for key in weights:
            if key not in scores:
                scores[key] = 0.5  # Neutral score
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        
        return {
            'component_scores': scores,
            'overall_score': overall_score,
            'validation_quality': self._interpret_score(overall_score)
        }
    
    def _interpret_score(self, score: float) -> str:
        """Interpret validation score"""
        
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def print_validation_report(self):
        """Print comprehensive validation report"""
        
        if not self.validation_results:
            print("No validation results available")
            return
        
        results = self.validation_results
        
        print("\n" + "="*70)
        print("SYNTHETIC DATA VALIDATION REPORT")
        print("="*70)
        
        # Overall score
        overall = results['overall_score']
        print(f"\nOverall Validation Score: {overall['overall_score']:.3f}")
        print(f"Validation Quality: {overall['validation_quality']}")
        
        # Component scores
        print(f"\nComponent Scores:")
        for component, score in overall['component_scores'].items():
            print(f"  {component.replace('_', ' ').title()}: {score:.3f}")
        
        # Summary statistics comparison
        summary = results['summary_statistics']
        perf = summary['performance_comparison']
        
        print(f"\nPerformance Comparison:")
        print(f"  Annual Return - Historical: {perf['hist_annual_return']:.2%}, Synthetic: {perf['synth_annual_return']:.2%}")
        print(f"  Annual Volatility - Historical: {perf['hist_annual_volatility']:.2%}, Synthetic: {perf['synth_annual_volatility']:.2%}")
        print(f"  Sharpe Ratio - Historical: {perf['hist_sharpe']:.2f}, Synthetic: {perf['synth_sharpe']:.2f}")
        
        # Risk metrics
        risk = results['risk_metrics']
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%) - Historical: {risk['var_metrics']['hist_var_95']:.3f}, Synthetic: {risk['var_metrics']['synth_var_95']:.3f}")
        print(f"  VaR (99%) - Historical: {risk['var_metrics']['hist_var_99']:.3f}, Synthetic: {risk['var_metrics']['synth_var_99']:.3f}")
        print(f"  Max Drawdown - Historical: {risk['drawdown_metrics']['hist_max_dd']:.2%}, Synthetic: {risk['drawdown_metrics']['synth_max_dd']:.2%}")
        
        # Distribution properties
        dist = results['distributional_properties']
        print(f"\nDistribution Properties:")
        print(f"  Mean Quantile Difference: {dist['quantile_comparison']['mean_quantile_diff']:.4f}")
        print(f"  Extreme Event Ratio: {dist['tail_behavior']['extreme_event_ratio']:.2f}")
        
        # Statistical tests
        ret_dist = results['return_distribution']
        print(f"\nStatistical Tests:")
        print(f"  Distributions Similar (KS Test): {ret_dist['statistical_tests']['distributions_similar']}")
        print(f"  KS Test p-value: {ret_dist['statistical_tests']['ks_pvalue']:.4f}")
        
        print("="*70)

    def _validate_leverage_effect(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate leverage effect: negative correlation between returns and volatility changes"""
        
        # Ensure historical data is loaded
        if self.historical_data is None:
            self.load_historical_data()
            
        if self.historical_data is None:
            # If still None, return default values
            return {
                'historical_leverage_correlation': np.nan,
                'synthetic_leverage_correlation': np.nan,
                'leverage_difference': np.nan,
                'leverage_effect_present': False
            }
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        def calculate_leverage_correlation(returns):
            """Calculate correlation between returns and next period volatility"""
            if len(returns) < 50:
                return np.nan
                
            # Calculate rolling volatility
            vol_series = returns.rolling(21).std()
            
            # Align returns with next period volatility
            returns_t = returns[:-1]
            vol_t_plus_1 = vol_series.shift(-1)[:-1]
            
            # Remove NaN values
            mask = ~(np.isnan(returns_t) | np.isnan(vol_t_plus_1))
            returns_clean = returns_t[mask]
            vol_clean = vol_t_plus_1[mask]
            
            if len(returns_clean) < 20:
                return np.nan
                
            # Calculate correlation
            correlation = np.corrcoef(returns_clean, vol_clean)[0, 1]
            return correlation
        
        hist_leverage = calculate_leverage_correlation(hist_returns)
        synth_leverage = calculate_leverage_correlation(synth_returns)
        
        return {
            'historical_leverage_correlation': hist_leverage,
            'synthetic_leverage_correlation': synth_leverage,
            'leverage_difference': abs(synth_leverage - hist_leverage) if not np.isnan(hist_leverage) and not np.isnan(synth_leverage) else np.nan,
            'leverage_effect_present': synth_leverage < -0.1 if not np.isnan(synth_leverage) else False
        }
    
    def _validate_fat_tails(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate fat tails in return distribution"""
        
        # Ensure historical data is loaded
        if self.historical_data is None:
            self.load_historical_data()
            
        if self.historical_data is None:
            # If still None, return default values
            return {
                'historical_kurtosis': np.nan,
                'synthetic_kurtosis': np.nan,
                'kurtosis_difference': np.nan,
                'both_fat_tails': False,
                'historical_extreme_frequency': np.nan,
                'synthetic_extreme_frequency': np.nan,
                'extreme_frequency_ratio': np.nan,
                'normality_rejected': {
                    'historical': False,
                    'synthetic': False
                }
            }
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Calculate kurtosis (measure of fat tails)
        hist_kurtosis = hist_returns.kurtosis()
        synth_kurtosis = synth_returns.kurtosis()
        
        # Jarque-Bera test for normality
        hist_jb_stat, hist_jb_pvalue = jarque_bera(hist_returns)
        synth_jb_stat, synth_jb_pvalue = jarque_bera(synth_returns)
        
        # Count extreme movements (> 3 std dev)
        hist_std = hist_returns.std()
        synth_std = synth_returns.std()
        
        hist_extreme_count = len(hist_returns[abs(hist_returns) > 3 * hist_std])
        synth_extreme_count = len(synth_returns[abs(synth_returns) > 3 * synth_std])
        
        hist_extreme_freq = hist_extreme_count / len(hist_returns)
        synth_extreme_freq = synth_extreme_count / len(synth_returns)
        
        return {
            'historical_kurtosis': hist_kurtosis,
            'synthetic_kurtosis': synth_kurtosis,
            'kurtosis_difference': abs(synth_kurtosis - hist_kurtosis),
            'both_fat_tails': hist_kurtosis > 3 and synth_kurtosis > 3,
            'historical_extreme_frequency': hist_extreme_freq,
            'synthetic_extreme_frequency': synth_extreme_freq,
            'extreme_frequency_ratio': synth_extreme_freq / hist_extreme_freq if hist_extreme_freq > 0 else np.nan,
            'normality_rejected': {
                'historical': hist_jb_pvalue < 0.05,
                'synthetic': synth_jb_pvalue < 0.05
            }
        }
    
    def _validate_volatility_clustering_detailed(self, synthetic_data: pd.DataFrame) -> Dict:
        """Detailed validation of volatility clustering"""
        
        # Ensure historical data is loaded
        if self.historical_data is None:
            self.load_historical_data()
            
        if self.historical_data is None:
            # If still None, return default values
            return {
                'arch_test': {
                    'historical_stat': np.nan,
                    'historical_pvalue': np.nan,
                    'synthetic_stat': np.nan,
                    'synthetic_pvalue': np.nan,
                    'both_clustering': False
                },
                'ljung_box_test': {
                    'historical_stat': np.nan,
                    'historical_pvalue': np.nan,
                    'synthetic_stat': np.nan,
                    'synthetic_pvalue': np.nan,
                    'both_clustering': False
                }
            }
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Test for ARCH effects (volatility clustering)
        def arch_test(returns, lags=5):
            """Engle's ARCH test for volatility clustering"""
            from scipy.stats import f
            
            # Squared returns
            squared_returns = returns**2
            
            # Regression: r²_t = α₀ + α₁*r²_{t-1} + ... + α_p*r²_{t-p} + ε_t
            n = len(squared_returns)
            
            # Create lagged variables
            X = np.ones((n - lags, lags + 1))
            for i in range(lags):
                X[:, i + 1] = squared_returns[lags - 1 - i:n - 1 - i]
            
            y = squared_returns[lags:]
            
            # OLS regression
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ y)
                y_pred = X @ beta
                residuals = y - y_pred
                
                # Calculate R²
                tss = np.sum((y - np.mean(y))**2)
                rss = np.sum(residuals**2)
                r_squared = 1 - rss / tss
                
                # F-statistic for ARCH test
                f_stat = (r_squared / lags) / ((1 - r_squared) / (n - lags - 1))
                
                # P-value
                p_value = 1 - f.cdf(f_stat, lags, n - lags - 1)
                
                return f_stat, p_value
            except:
                return np.nan, np.nan
        
        hist_arch_stat, hist_arch_pvalue = arch_test(hist_returns)
        synth_arch_stat, synth_arch_pvalue = arch_test(synth_returns)
        
        # Ljung-Box test on squared returns
        from scipy.stats import chi2
        
        def ljung_box_test(series, lags=10):
            """Ljung-Box test for serial correlation"""
            n = len(series)
            
            # Calculate autocorrelations
            autocorrs = []
            for lag in range(1, lags + 1):
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
                else:
                    autocorrs.append(0)
            
            # Calculate test statistic
            lb_stat = n * (n + 2) * sum([(autocorr**2) / (n - k) for k, autocorr in enumerate(autocorrs, 1)])
            
            # P-value
            p_value = 1 - chi2.cdf(lb_stat, lags)
            
            return lb_stat, p_value
        
        hist_lb_stat, hist_lb_pvalue = ljung_box_test(hist_returns**2)
        synth_lb_stat, synth_lb_pvalue = ljung_box_test(synth_returns**2)
        
        return {
            'arch_test': {
                'historical_stat': hist_arch_stat,
                'historical_pvalue': hist_arch_pvalue,
                'synthetic_stat': synth_arch_stat,
                'synthetic_pvalue': synth_arch_pvalue,
                'both_clustering': hist_arch_pvalue < 0.05 and synth_arch_pvalue < 0.05
            },
            'ljung_box_test': {
                'historical_stat': hist_lb_stat,
                'historical_pvalue': hist_lb_pvalue,
                'synthetic_stat': synth_lb_stat,
                'synthetic_pvalue': synth_lb_pvalue,
                'both_clustering': hist_lb_pvalue < 0.05 and synth_lb_pvalue < 0.05
            }
        }
    
    def _validate_stylized_facts(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate key stylized facts of financial returns"""
        
        # Ensure historical data is loaded
        if self.historical_data is None:
            self.load_historical_data()
            
        if self.historical_data is None:
            # If still None, return default values with synthetic-only validation
            synth_returns = synthetic_data['simple_return'].dropna()
            
            return {
                'heavy_tails': {
                    'historical': False,
                    'synthetic': synth_returns.kurtosis() > 3,
                    'both_satisfied': False
                },
                'negative_skewness': {
                    'historical': False,
                    'synthetic': synth_returns.skew() < 0,
                    'both_satisfied': False
                },
                'volatility_clustering': {
                    'historical': False,
                    'synthetic': False,
                    'both_satisfied': False
                },
                'leverage_effect': {
                    'historical': False,
                    'synthetic': False,
                    'both_satisfied': False
                },
                'no_return_autocorr': {
                    'historical': False,
                    'synthetic': False,
                    'both_satisfied': False
                },
                'overall_score': 0.0
            }
        
        hist_returns = self.historical_data['Returns'].dropna()
        synth_returns = synthetic_data['simple_return'].dropna()
        
        # Stylized facts checklist
        stylized_facts = {}
        
        # 1. Heavy tails (kurtosis > 3)
        stylized_facts['heavy_tails'] = {
            'historical': hist_returns.kurtosis() > 3,
            'synthetic': synth_returns.kurtosis() > 3,
            'both_satisfied': hist_returns.kurtosis() > 3 and synth_returns.kurtosis() > 3
        }
        
        # 2. Negative skewness
        stylized_facts['negative_skewness'] = {
            'historical': hist_returns.skew() < 0,
            'synthetic': synth_returns.skew() < 0,
            'both_satisfied': hist_returns.skew() < 0 and synth_returns.skew() < 0
        }
        
        # 3. Volatility clustering (ARCH effects)
        try:
            arch_test_results = self._validate_volatility_clustering_detailed(synthetic_data)
            stylized_facts['volatility_clustering'] = {
                'historical': arch_test_results['arch_test']['historical_pvalue'] < 0.05,
                'synthetic': arch_test_results['arch_test']['synthetic_pvalue'] < 0.05,
                'both_satisfied': arch_test_results['arch_test']['both_clustering']
            }
        except:
            stylized_facts['volatility_clustering'] = {
                'historical': False,
                'synthetic': False,
                'both_satisfied': False
            }
        
        # 4. Leverage effect
        try:
            leverage_results = self._validate_leverage_effect(synthetic_data)
            stylized_facts['leverage_effect'] = {
                'historical': leverage_results['historical_leverage_correlation'] < -0.1,
                'synthetic': leverage_results['synthetic_leverage_correlation'] < -0.1,
                'both_satisfied': (leverage_results['historical_leverage_correlation'] < -0.1 and 
                                 leverage_results['synthetic_leverage_correlation'] < -0.1)
            }
        except:
            stylized_facts['leverage_effect'] = {
                'historical': False,
                'synthetic': False,
                'both_satisfied': False
            }
        
        # 5. Absence of autocorrelation in returns
        try:
            def ljung_box_test_local(series, lags=10):
                """Ljung-Box test for serial correlation"""
                n = len(series)
                
                # Calculate autocorrelations
                autocorrs = []
                for lag in range(1, lags + 1):
                    if len(series) > lag:
                        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        if not np.isnan(corr):
                            autocorrs.append(corr)
                        else:
                            autocorrs.append(0)
                
                # Calculate test statistic
                lb_stat = n * (n + 2) * sum([(autocorr**2) / (n - k) for k, autocorr in enumerate(autocorrs, 1)])
                
                # P-value
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(lb_stat, len(autocorrs))
                
                return lb_stat, p_value
            
            lb_returns_hist = ljung_box_test_local(hist_returns)
            lb_returns_synth = ljung_box_test_local(synth_returns)
            stylized_facts['no_return_autocorr'] = {
                'historical': lb_returns_hist[1] > 0.05,
                'synthetic': lb_returns_synth[1] > 0.05,
                'both_satisfied': lb_returns_hist[1] > 0.05 and lb_returns_synth[1] > 0.05
            }
        except:
            stylized_facts['no_return_autocorr'] = {
                'historical': False,
                'synthetic': False,
                'both_satisfied': False
            }
        
        # Calculate overall stylized facts score
        total_facts = len(stylized_facts)
        satisfied_facts = sum([fact['both_satisfied'] for fact in stylized_facts.values()])
        stylized_facts['overall_score'] = satisfied_facts / total_facts
        
        return stylized_facts

if __name__ == "__main__":
    # Example usage
    print("Testing Validation Module...")
    
    # This would typically be called with actual synthetic data
    # from synthetic_generator import SyntheticStockDataGenerator
    # 
    # generator = SyntheticStockDataGenerator(random_seed=42)
    # synthetic_data = generator.generate_synthetic_data(n_years=10)
    # 
    # validator = SyntheticDataValidator()
    # results = validator.validate_synthetic_data(synthetic_data)
    # validator.print_validation_report()
    
    print("Validation module ready for use") 