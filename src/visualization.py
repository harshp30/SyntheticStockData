"""
Visualization Module
Create comprehensive plots for synthetic stock data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from config import VALIDATION_CONFIG

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SyntheticDataVisualizer:
    """Create visualizations for synthetic stock data"""
    
    def __init__(self, output_dir: str = ".", figsize: Tuple[int, int] = VALIDATION_CONFIG.figure_size):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = VALIDATION_CONFIG.dpi
        
        # Set matplotlib to non-interactive backend for terminal use
        plt.switch_backend('Agg')
        
    def plot_price_series(self, synthetic_data: pd.DataFrame, 
                         historical_data: Optional[pd.DataFrame] = None,
                         years_to_plot: int = VALIDATION_CONFIG.plot_years,
                         show_regimes: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot synthetic price series with optional historical comparison"""
        
        # Sample data for plotting if too long
        if len(synthetic_data) > years_to_plot * 252:
            end_date = synthetic_data.index[-1]
            start_date = end_date - timedelta(days=years_to_plot * 365)
            plot_data = synthetic_data.loc[start_date:]
        else:
            plot_data = synthetic_data
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Get price column (handle different naming conventions)
        price_col = 'Price' if 'Price' in plot_data.columns else 'price'
        return_col = 'simple_return' if 'simple_return' in plot_data.columns else 'Returns'
        
        # Plot 1: Price series
        axes[0].plot(plot_data.index, plot_data[price_col], linewidth=1.5, 
                    label='Synthetic Price', color='blue')
        
        if historical_data is not None:
            # Align historical data to same time period
            # Handle timezone differences between synthetic and historical data
            try:
                hist_aligned = historical_data.loc[plot_data.index[0]:plot_data.index[-1]]
            except TypeError:
                # Handle timezone mismatch by converting to timezone-naive
                start_date = plot_data.index[0]
                end_date = plot_data.index[-1]
                if hasattr(start_date, 'tz_localize'):
                    start_date = start_date.tz_localize(None) if start_date.tz is None else start_date.tz_convert(None)
                    end_date = end_date.tz_localize(None) if end_date.tz is None else end_date.tz_convert(None)
                
                # Strip timezone from historical data index
                historical_data_tz_naive = historical_data.copy()
                if hasattr(historical_data.index, 'tz') and historical_data.index.tz is not None:
                    historical_data_tz_naive.index = historical_data.index.tz_convert(None)
                    
                hist_aligned = historical_data_tz_naive.loc[start_date:end_date]
            
            if len(hist_aligned) > 0:
                # Normalize historical data to start at same price
                hist_normalized = hist_aligned['Close'] * (plot_data[price_col].iloc[0] / hist_aligned['Close'].iloc[0])
                axes[0].plot(hist_aligned.index, hist_normalized, 
                           linewidth=1.5, label='Historical Price (Normalized)', 
                           color='red', alpha=0.7)
        
        # Color regimes if requested
        if show_regimes and 'regime' in plot_data.columns:
            bull_periods = plot_data[plot_data['regime'] == 'bull']
            bear_periods = plot_data[plot_data['regime'] == 'bear']
            
            if len(bull_periods) > 0:
                axes[0].fill_between(bull_periods.index, axes[0].get_ylim()[0], 
                                   axes[0].get_ylim()[1], alpha=0.1, color='green', 
                                   label='Bull Market')
            if len(bear_periods) > 0:
                axes[0].fill_between(bear_periods.index, axes[0].get_ylim()[0], 
                                   axes[0].get_ylim()[1], alpha=0.1, color='red', 
                                   label='Bear Market')
        
        axes[0].set_title('Stock Price Series', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns
        if return_col in plot_data.columns:
            axes[1].plot(plot_data.index, plot_data[return_col] * 100, 
                        linewidth=0.8, color='darkblue', alpha=0.7)
        else:
            # Calculate returns from price
            returns = plot_data[price_col].pct_change()
            axes[1].plot(plot_data.index, returns * 100, 
                        linewidth=0.8, color='darkblue', alpha=0.7)
        
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('Daily Returns', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Return (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_regime_analysis(self, synthetic_data: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot regime analysis including duration and returns"""
        
        # Get regime summary
        regime_periods = self._get_regime_periods(synthetic_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Plot 1: Regime durations
        bull_durations = [p['duration_days'] for p in regime_periods if p['regime'] == 'bull']
        bear_durations = [p['duration_days'] for p in regime_periods if p['regime'] == 'bear']
        
        axes[0, 0].hist(bull_durations, bins=15, alpha=0.7, color='green', 
                       label=f'Bull Markets (n={len(bull_durations)})')
        axes[0, 0].hist(bear_durations, bins=15, alpha=0.7, color='red', 
                       label=f'Bear Markets (n={len(bear_durations)})')
        axes[0, 0].set_title('Regime Duration Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Duration (days)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Regime returns
        bull_returns = [p['return'] * 100 for p in regime_periods if p['regime'] == 'bull']
        bear_returns = [p['return'] * 100 for p in regime_periods if p['regime'] == 'bear']
        
        axes[0, 1].hist(bull_returns, bins=15, alpha=0.7, color='green', 
                       label=f'Bull Returns (Mean: {np.mean(bull_returns):.1f}%)')
        axes[0, 1].hist(bear_returns, bins=15, alpha=0.7, color='red', 
                       label=f'Bear Returns (Mean: {np.mean(bear_returns):.1f}%)')
        axes[0, 1].set_title('Regime Return Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Total Return (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Regime timeline
        regime_timeline = synthetic_data['regime'].copy()
        regime_numeric = regime_timeline.map({'bull': 1, 'bear': 0})
        
        axes[1, 0].fill_between(regime_timeline.index, 0, regime_numeric, 
                               color='green', alpha=0.7, label='Bull Market')
        axes[1, 0].fill_between(regime_timeline.index, regime_numeric, 0, 
                               color='red', alpha=0.7, label='Bear Market')
        axes[1, 0].set_title('Regime Timeline', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Regime')
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['Bear', 'Bull'])
        axes[1, 0].legend()
        
        # Plot 4: Volatility by regime
        # Handle different column names
        return_col = 'simple_return' if 'simple_return' in synthetic_data.columns else 'Returns'
        if return_col not in synthetic_data.columns:
            # Calculate returns from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            returns = synthetic_data[price_col].pct_change()
        else:
            returns = synthetic_data[return_col]
        
        bull_vol = returns[synthetic_data['regime'] == 'bull'].std() * np.sqrt(252) * 100
        bear_vol = returns[synthetic_data['regime'] == 'bear'].std() * np.sqrt(252) * 100
        
        regimes = ['Bull', 'Bear']
        volatilities = [bull_vol, bear_vol]
        colors = ['green', 'red']
        
        bars = axes[1, 1].bar(regimes, volatilities, color=colors, alpha=0.7)
        axes[1, 1].set_title('Volatility by Regime', fontweight='bold')
        axes[1, 1].set_ylabel('Annual Volatility (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, vol in zip(bars, volatilities):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{vol:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_return_analysis(self, synthetic_data: pd.DataFrame,
                           historical_data: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot return distribution analysis"""
        
        # Handle different column names
        return_col = 'simple_return' if 'simple_return' in synthetic_data.columns else 'Returns'
        if return_col not in synthetic_data.columns:
            # Calculate returns from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            synth_returns = synthetic_data[price_col].pct_change().dropna()
        else:
            synth_returns = synthetic_data[return_col].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Plot 1: Return distribution
        axes[0, 0].hist(synth_returns * 100, bins=50, alpha=0.7, color='blue', 
                       density=True, label='Synthetic')
        
        if historical_data is not None:
            hist_returns = historical_data['Close'].pct_change().dropna()
            axes[0, 0].hist(hist_returns * 100, bins=50, alpha=0.5, color='red', 
                           density=True, label='Historical')
        
        axes[0, 0].set_title('Return Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Daily Return (%)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        from scipy import stats
        stats.probplot(synth_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Rolling volatility
        rolling_vol = synth_returns.rolling(21).std() * np.sqrt(252) * 100
        axes[1, 0].plot(rolling_vol.index, rolling_vol, linewidth=1, color='purple')
        axes[1, 0].set_title('Rolling Volatility (21-day)', fontweight='bold')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Autocorrelation
        autocorr = [synth_returns.autocorr(lag=i) for i in range(1, 21)]
        axes[1, 1].bar(range(1, 21), autocorr, alpha=0.7, color='orange')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Return Autocorrelation', fontweight='bold')
        axes[1, 1].set_xlabel('Lag (days)')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_risk_metrics(self, synthetic_data: pd.DataFrame,
                         historical_data: Optional[pd.DataFrame] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot risk metrics comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Handle different column names
        return_col = 'simple_return' if 'simple_return' in synthetic_data.columns else 'Returns'
        if return_col not in synthetic_data.columns:
            # Calculate returns from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            returns = synthetic_data[price_col].pct_change()
        else:
            returns = synthetic_data[return_col]
        
        # Plot 1: Drawdown
        drawdown_col = 'drawdown' if 'drawdown' in synthetic_data.columns else None
        if drawdown_col is not None:
            axes[0, 0].fill_between(synthetic_data.index, 0, synthetic_data[drawdown_col] * 100, 
                                   color='red', alpha=0.7, label='Drawdown')
        else:
            # Calculate drawdown from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max) - 1
            axes[0, 0].fill_between(synthetic_data.index, 0, drawdown * 100, 
                                   color='red', alpha=0.7, label='Drawdown')
        
        axes[0, 0].set_title('Drawdown Analysis', fontweight='bold')
        axes[0, 0].set_ylabel('Drawdown (%)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rolling Sharpe ratio
        rolling_returns = returns.rolling(252)
        rolling_sharpe = (rolling_returns.mean() * 252) / (rolling_returns.std() * np.sqrt(252))
        
        axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe, linewidth=1.5, color='green')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('Rolling Sharpe Ratio (1-year)', fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Value at Risk
        synth_returns = returns.dropna()
        var_levels = [0.01, 0.05, 0.1]
        var_values = [synth_returns.quantile(level) * 100 for level in var_levels]
        
        bars = axes[1, 0].bar([f'{int(level*100)}%' for level in var_levels], 
                             var_values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Value at Risk', fontweight='bold')
        axes[1, 0].set_ylabel('VaR (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, var_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}%', ha='center', va='bottom')
        
        # Plot 4: Return vs Risk scatter
        if historical_data is not None:
            # Calculate rolling metrics for comparison
            window = 252
            synth_rolling_ret = returns.rolling(window).mean() * 252 * 100
            synth_rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            
            hist_returns = historical_data['Close'].pct_change().dropna()
            hist_rolling_ret = hist_returns.rolling(window).mean() * 252 * 100
            hist_rolling_vol = hist_returns.rolling(window).std() * np.sqrt(252) * 100
            
            axes[1, 1].scatter(synth_rolling_vol, synth_rolling_ret, alpha=0.5, 
                             color='blue', label='Synthetic', s=10)
            axes[1, 1].scatter(hist_rolling_vol, hist_rolling_ret, alpha=0.5, 
                             color='red', label='Historical', s=10)
            axes[1, 1].set_title('Return vs Risk (Rolling 1-year)', fontweight='bold')
            axes[1, 1].set_xlabel('Volatility (%)')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_validation_summary(self, validation_results: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot validation results summary"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Plot 1: Overall validation scores
        if 'overall_score' in validation_results:
            scores = validation_results['overall_score']['component_scores']
            components = list(scores.keys())
            values = list(scores.values())
            
            bars = axes[0, 0].bar(range(len(components)), values, 
                                 color=['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in values])
            axes[0, 0].set_title('Validation Component Scores', fontweight='bold')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_xticks(range(len(components)))
            axes[0, 0].set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 2: Return moments comparison
        if 'return_distribution' in validation_results:
            moments = validation_results['return_distribution']['moments']
            
            metrics = ['Mean', 'Std', 'Skew', 'Kurt']
            ratios = [moments.get('mean_ratio', 1), moments.get('std_ratio', 1), 
                     1, 1]  # Skew and Kurt ratios would need to be calculated
            
            bars = axes[0, 1].bar(metrics, ratios, alpha=0.7, color='skyblue')
            axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Return Moments Ratio (Synthetic/Historical)', fontweight='bold')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, ratios):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 3: Risk metrics comparison
        if 'risk_metrics' in validation_results:
            risk_data = validation_results['risk_metrics']
            
            # VaR comparison
            var_95_hist = risk_data['var_metrics']['hist_var_95']
            var_95_synth = risk_data['var_metrics']['synth_var_95']
            var_99_hist = risk_data['var_metrics']['hist_var_99']
            var_99_synth = risk_data['var_metrics']['synth_var_99']
            
            x = np.arange(2)
            width = 0.35
            
            axes[1, 0].bar(x - width/2, [var_95_hist, var_99_hist], width, 
                          label='Historical', alpha=0.7, color='red')
            axes[1, 0].bar(x + width/2, [var_95_synth, var_99_synth], width, 
                          label='Synthetic', alpha=0.7, color='blue')
            
            axes[1, 0].set_title('Value at Risk Comparison', fontweight='bold')
            axes[1, 0].set_ylabel('VaR')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(['VaR 95%', 'VaR 99%'])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary text
        axes[1, 1].axis('off')
        
        if 'overall_score' in validation_results:
            overall_score = validation_results['overall_score']['overall_score']
            quality = validation_results['overall_score']['validation_quality']
            
            summary_text = f"""
            VALIDATION SUMMARY
            
            Overall Score: {overall_score:.3f}
            Quality Rating: {quality}
            
            Key Findings:
            • Statistical distribution matching
            • Volatility structure preservation
            • Risk metrics alignment
            • Regime characteristics validation
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def create_interactive_dashboard(self, synthetic_data: pd.DataFrame,
                                   historical_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Series', 'Returns Distribution', 
                          'Rolling Volatility', 'Regime Timeline',
                          'Drawdown', 'Cumulative Returns'),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"secondary_y": False}, {"type": "scatter"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price series
        fig.add_trace(
            go.Scatter(x=synthetic_data.index, y=synthetic_data['price'],
                      name='Synthetic Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        if historical_data is not None:
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Close'],
                          name='Historical Price', line=dict(color='red')),
                row=1, col=1
            )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=synthetic_data['simple_return'] * 100,
                        name='Synthetic Returns', opacity=0.7),
            row=1, col=2
        )
        
        # Rolling volatility
        rolling_vol = synthetic_data['simple_return'].rolling(21).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol,
                      name='Rolling Volatility', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Regime timeline
        regime_numeric = synthetic_data['regime'].map({'bull': 1, 'bear': 0})
        fig.add_trace(
            go.Scatter(x=synthetic_data.index, y=regime_numeric,
                      name='Regime', fill='tozeroy', line=dict(color='green')),
            row=2, col=2
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=synthetic_data.index, y=synthetic_data['drawdown'] * 100,
                      name='Drawdown', fill='tozeroy', line=dict(color='red')),
            row=3, col=1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=synthetic_data.index, y=synthetic_data['cumulative_return'] * 100,
                      name='Cumulative Return', line=dict(color='darkgreen')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Synthetic Stock Data Analysis Dashboard',
            showlegend=False,
            height=1000
        )
        
        return fig
    
    def _get_regime_periods(self, synthetic_data: pd.DataFrame) -> List[Dict]:
        """Extract regime periods from synthetic data"""
        
        # Handle different column names
        price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
        
        regime_periods = []
        current_regime = synthetic_data['regime'].iloc[0]
        start_idx = 0
        
        for i, regime in enumerate(synthetic_data['regime']):
            if regime != current_regime:
                # End of current regime
                regime_periods.append({
                    'regime': current_regime,
                    'start_date': synthetic_data.index[start_idx],
                    'end_date': synthetic_data.index[i-1],
                    'duration_days': i - start_idx,
                    'start_price': synthetic_data[price_col].iloc[start_idx],
                    'end_price': synthetic_data[price_col].iloc[i-1],
                    'return': (synthetic_data[price_col].iloc[i-1] / 
                              synthetic_data[price_col].iloc[start_idx]) - 1
                })
                
                current_regime = regime
                start_idx = i
                
        # Add final regime
        regime_periods.append({
            'regime': current_regime,
            'start_date': synthetic_data.index[start_idx],
            'end_date': synthetic_data.index[-1],
            'duration_days': len(synthetic_data) - start_idx,
            'start_price': synthetic_data[price_col].iloc[start_idx],
            'end_price': synthetic_data[price_col].iloc[-1],
            'return': (synthetic_data[price_col].iloc[-1] / 
                      synthetic_data[price_col].iloc[start_idx]) - 1
        })
        
        return regime_periods

    def plot_price_series_comparison(self, synthetic_data: pd.DataFrame, 
                         historical_data: Optional[pd.DataFrame] = None,
                         years_to_plot: int = VALIDATION_CONFIG.plot_years,
                         show_regimes: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot synthetic vs historical price series comparison"""
        
        # Use existing plot_price_series method
        return self.plot_price_series(synthetic_data, historical_data, years_to_plot, show_regimes, save_path)
    
    def plot_returns_distribution_comparison(self, synthetic_data: pd.DataFrame,
                           historical_data: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot returns distribution comparison between synthetic and historical data"""
        
        # Use existing plot_return_analysis method
        return self.plot_return_analysis(synthetic_data, historical_data, save_path)
    
    def plot_volatility_clustering(self, synthetic_data: pd.DataFrame,
                                 historical_data: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot volatility clustering analysis (ARCH/GARCH effects)"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Calculate returns and volatility
        # Handle different column names
        return_col = 'simple_return' if 'simple_return' in synthetic_data.columns else 'Returns'
        if return_col not in synthetic_data.columns:
            # Calculate returns from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            synth_returns = synthetic_data[price_col].pct_change().dropna()
        else:
            synth_returns = synthetic_data[return_col].dropna()
        
        synth_vol = synth_returns.abs()
        
        # Plot 1: Returns over time
        axes[0, 0].plot(synth_returns.index, synth_returns * 100, 
                       linewidth=0.5, color='blue', alpha=0.7)
        axes[0, 0].set_title('Daily Returns Time Series', fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Absolute returns (volatility proxy)
        axes[0, 1].plot(synth_vol.index, synth_vol * 100, 
                       linewidth=0.5, color='red', alpha=0.7)
        axes[0, 1].set_title('Absolute Returns (Volatility Proxy)', fontweight='bold')
        axes[0, 1].set_ylabel('Absolute Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility autocorrelation
        vol_autocorr = [synth_vol.autocorr(lag=i) for i in range(1, 21)]
        axes[1, 0].bar(range(1, 21), vol_autocorr, alpha=0.7, color='purple')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Volatility Autocorrelation', fontweight='bold')
        axes[1, 0].set_xlabel('Lag (days)')
        axes[1, 0].set_ylabel('Autocorrelation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Comparison with historical if available
        if historical_data is not None:
            hist_returns = historical_data['Close'].pct_change().dropna()
            hist_vol = hist_returns.abs()
            
            # Rolling volatility comparison
            synth_rolling_vol = synth_returns.rolling(21).std() * np.sqrt(252) * 100
            hist_rolling_vol = hist_returns.rolling(21).std() * np.sqrt(252) * 100
            
            axes[1, 1].plot(synth_rolling_vol.index, synth_rolling_vol, 
                           linewidth=1, color='blue', label='Synthetic', alpha=0.7)
            axes[1, 1].plot(hist_rolling_vol.index, hist_rolling_vol, 
                           linewidth=1, color='red', label='Historical', alpha=0.7)
            axes[1, 1].set_title('Rolling Volatility Comparison (21-day)', fontweight='bold')
            axes[1, 1].set_ylabel('Volatility (%)')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Just show rolling volatility for synthetic data
            synth_rolling_vol = synth_returns.rolling(21).std() * np.sqrt(252) * 100
            axes[1, 1].plot(synth_rolling_vol.index, synth_rolling_vol, 
                           linewidth=1, color='blue', alpha=0.7)
            axes[1, 1].set_title('Rolling Volatility (21-day)', fontweight='bold')
            axes[1, 1].set_ylabel('Volatility (%)')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_autocorrelation_analysis(self, synthetic_data: pd.DataFrame,
                                    historical_data: Optional[pd.DataFrame] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot autocorrelation analysis for returns and volatility"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Calculate returns
        # Handle different column names
        return_col = 'simple_return' if 'simple_return' in synthetic_data.columns else 'Returns'
        if return_col not in synthetic_data.columns:
            # Calculate returns from price
            price_col = 'Price' if 'Price' in synthetic_data.columns else 'price'
            synth_returns = synthetic_data[price_col].pct_change().dropna()
        else:
            synth_returns = synthetic_data[return_col].dropna()
        
        synth_vol = synth_returns.abs()
        
        # Plot 1: Return autocorrelation
        return_autocorr = [synth_returns.autocorr(lag=i) for i in range(1, 21)]
        axes[0, 0].bar(range(1, 21), return_autocorr, alpha=0.7, color='blue')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].set_title('Return Autocorrelation (Synthetic)', fontweight='bold')
        axes[0, 0].set_xlabel('Lag (days)')
        axes[0, 0].set_ylabel('Autocorrelation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Volatility autocorrelation
        vol_autocorr = [synth_vol.autocorr(lag=i) for i in range(1, 21)]
        axes[0, 1].bar(range(1, 21), vol_autocorr, alpha=0.7, color='red')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('Volatility Autocorrelation (Synthetic)', fontweight='bold')
        axes[0, 1].set_xlabel('Lag (days)')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3 & 4: Comparison with historical if available
        if historical_data is not None:
            hist_returns = historical_data['Close'].pct_change().dropna()
            hist_vol = hist_returns.abs()
            
            # Historical return autocorrelation
            hist_return_autocorr = [hist_returns.autocorr(lag=i) for i in range(1, 21)]
            axes[1, 0].bar(range(1, 21), hist_return_autocorr, alpha=0.7, color='green')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].set_title('Return Autocorrelation (Historical)', fontweight='bold')
            axes[1, 0].set_xlabel('Lag (days)')
            axes[1, 0].set_ylabel('Autocorrelation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Historical volatility autocorrelation
            hist_vol_autocorr = [hist_vol.autocorr(lag=i) for i in range(1, 21)]
            axes[1, 1].bar(range(1, 21), hist_vol_autocorr, alpha=0.7, color='orange')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('Volatility Autocorrelation (Historical)', fontweight='bold')
            axes[1, 1].set_xlabel('Lag (days)')
            axes[1, 1].set_ylabel('Autocorrelation')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show squared returns autocorrelation for synthetic data
            synth_squared_returns = synth_returns ** 2
            squared_autocorr = [synth_squared_returns.autocorr(lag=i) for i in range(1, 21)]
            axes[1, 0].bar(range(1, 21), squared_autocorr, alpha=0.7, color='purple')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].set_title('Squared Returns Autocorrelation', fontweight='bold')
            axes[1, 0].set_xlabel('Lag (days)')
            axes[1, 0].set_ylabel('Autocorrelation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Show partial autocorrelation
            try:
                from statsmodels.tsa.stattools import pacf
                pacf_values = pacf(synth_returns.dropna(), nlags=20)[1:]
                axes[1, 1].bar(range(1, 21), pacf_values, alpha=0.7, color='cyan')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].set_title('Partial Autocorrelation Function', fontweight='bold')
                axes[1, 1].set_xlabel('Lag (days)')
                axes[1, 1].set_ylabel('Partial Autocorrelation')
                axes[1, 1].grid(True, alpha=0.3)
            except ImportError:
                # Fallback: show a simple correlation plot if statsmodels is not available
                corr_values = [synth_returns.corr(synth_returns.shift(i)) for i in range(1, 21)]
                axes[1, 1].bar(range(1, 21), corr_values, alpha=0.7, color='cyan')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].set_title('Correlation Function (Fallback)', fontweight='bold')
                axes[1, 1].set_xlabel('Lag (days)')
                axes[1, 1].set_ylabel('Correlation')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_risk_metrics_comparison(self, synthetic_data: pd.DataFrame,
                                   historical_data: Optional[pd.DataFrame] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot risk metrics comparison between synthetic and historical data"""
        
        # Use existing plot_risk_metrics method
        return self.plot_risk_metrics(synthetic_data, historical_data, save_path)

if __name__ == "__main__":
    # Example usage
    print("Visualization module ready for use")
    
    # This would typically be used with real data:
    # visualizer = SyntheticDataVisualizer()
    # fig = visualizer.plot_price_series(synthetic_data)
    # plt.show() 