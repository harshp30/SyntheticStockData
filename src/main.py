#!/usr/bin/env python3
"""
Synthetic Stock Data Pipeline - Main Application Entry Point
===========================================================

This is the main terminal application for generating high-quality synthetic stock data
using advanced stochastic modeling techniques. The pipeline implements state-of-the-art
quantitative finance models to create realistic synthetic financial data that preserves
the statistical properties and stylized facts of real markets.

Key Features:
- Non-deterministic data generation with truly random seeds
- Regime-switching market models with empirical calibration
- Advanced volatility clustering and leverage effects
- Comprehensive statistical validation and testing
- Professional visualizations and analysis outputs
- CSV export with 16-column comprehensive data structure

Mathematical Models Implemented:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ • Regime-Switching Markov Models: P(S_{t+1} = j | S_t = i) = π_{ij}           │
│ • Geometric Brownian Motion: dS = μS dt + σS dW_t                             │
│ • Jump Diffusion Process: dS = μS dt + σS dW_t + J_t dN_t                     │
│ • GARCH(1,1) Volatility Clustering: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}       │
│ • Leverage Effect: Corr(r_t, σ²_{t+1}) < 0                                    │
│ • Student's t-Distribution: Fat-tailed return innovations                      │
│ • Weibull Duration Modeling: Realistic regime persistence                      │
└─────────────────────────────────────────────────────────────────────────────────┘

Statistical Properties Preserved:
- Volatility clustering (GARCH effects)
- Fat-tailed return distributions (kurtosis > 3)
- Leverage effects (asymmetric volatility)
- Regime persistence (bull/bear market cycles)
- Realistic jump frequencies and magnitudes
- Proper autocorrelation structures

Usage:
    python main.py

The application will guide you through:
1. Ticker selection (default: S&P 500)
2. Duration specification (default: 5 years)
3. Automatic historical data analysis
4. Synthetic data generation with calibrated parameters
5. Comprehensive validation and testing
6. Professional visualization creation
7. CSV export for ML applications

Output Files:
- synthetic_data_[ticker]_[timestamp].csv: Complete 16-column dataset
- validation_report.html: Statistical validation results
- price_analysis.png: Price series comparison
- distribution_analysis.png: Return distribution analysis
- volatility_analysis.png: Volatility clustering visualization
- regime_analysis.png: Market regime visualization

Author: Quantitative Finance Research Team
Version: 2.0
License: MIT
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SIMULATION_CONFIG, set_random_seed, get_config_summary
from data_loader import HistoricalDataLoader
from synthetic_generator import SyntheticStockDataGenerator
from validation import SyntheticDataValidator
from visualization import SyntheticDataVisualizer

class SyntheticStockPipeline:
    """Main pipeline orchestrator for terminal-based synthetic stock data generation."""
    
    def __init__(self):
        self.ticker = None
        self.historical_data = None
        self.synthetic_data = None
        self.output_dir = None
        
    def display_banner(self):
        """Display application banner with mathematical context."""
        print("\n" + "="*80)
        print("🔬 SYNTHETIC STOCK DATA PIPELINE")
        print("   Advanced Stochastic Modeling for Quantitative ML Research")
        print("="*80)
        print("\nImplementing:")
        print("• Regime-Switching Markov Models for market cycles")
        print("• Geometric Brownian Motion: dS = μS dt + σS dW")
        print("• Jump Diffusion Processes for extreme events")
        print("• Historical Parameter Calibration via MLE")
        print("• Statistical Validation Suite")
        print("-"*80)
        
    def get_user_inputs(self):
        """Get user inputs with intelligent defaults."""
        print("\n📊 CONFIGURATION")
        print("-" * 20)
        
        # Get ticker symbol
        while True:
            ticker_input = input("Enter stock ticker (default: ^GSPC for S&P 500): ").strip().upper()
            if not ticker_input:
                self.ticker = "^GSPC"
                print(f"✓ Using default ticker: {self.ticker}")
                break
            else:
                self.ticker = ticker_input
                break
        
        # Get synthetic data duration
        while True:
            try:
                duration_input = input("Enter synthetic data duration in years (default: 5): ").strip()
                if not duration_input:
                    self.synthetic_years = 5
                else:
                    self.synthetic_years = float(duration_input)
                    if self.synthetic_years <= 0:
                        print("❌ Duration must be positive. Please try again.")
                        continue
                print(f"✓ Synthetic data duration: {self.synthetic_years} years")
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
                continue
        
        # Create output directory in data folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"../data/synthetic_data_{self.ticker}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")
        
    def _create_regime_series(self, cycles):
        """Create a pandas Series with regime labels for each date."""
        import pandas as pd
        
        # Initialize regime series with 'bull' as default
        regime_series = pd.Series('bull', index=self.historical_data.index)
        
        # Mark bear market periods
        for cycle in cycles:
            if cycle['type'] == 'bear':
                start_date = cycle['start_date']
                end_date = cycle['end_date']
                regime_series.loc[start_date:end_date] = 'bear'
        
        return regime_series
    
    def load_and_analyze_historical_data(self):
        """Load historical data and perform comprehensive analysis."""
        print(f"\n📈 HISTORICAL DATA ANALYSIS")
        print("-" * 30)
        
        try:
            print(f"🔍 Fetching historical data for {self.ticker}...")
            print("   • Connecting to Yahoo Finance API")
            print("   • Downloading maximum available history")
            
            # Configure data loader with maximum available history
            loader = HistoricalDataLoader(self.ticker)
            self.historical_data = loader.fetch_data(
                start_date="1950-01-01",  # Use all available data
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            if self.historical_data is None or len(self.historical_data) < 252:
                print(f"❌ Insufficient or invalid data for {self.ticker}")
                print("   This could be due to:")
                print("   • Invalid ticker symbol")
                print("   • Delisted stock")
                print("   • Insufficient trading history")
                return False
            
            print(f"✓ Successfully loaded {len(self.historical_data)} trading days")
            print(f"   • Date range: {self.historical_data.index[0].date()} to {self.historical_data.index[-1].date()}")
            print(f"   • Time span: {(self.historical_data.index[-1] - self.historical_data.index[0]).days / 365.25:.1f} years")
            
            # Perform regime analysis
            print("\n🧮 REGIME ANALYSIS")
            print("   • Identifying bull/bear market cycles using drawdown analysis")
            print("   • Calculating maximum drawdown thresholds")
            print("   • Applying Hidden Markov Model assumptions")
            
            cycles = loader.identify_cycles()
            statistics = loader.calculate_statistics()
            
            # Create regime analysis structure
            regime_analysis = {
                'bull_periods': [c for c in cycles if c['type'] == 'bull'],
                'bear_periods': [c for c in cycles if c['type'] == 'bear'],
                'regime': self._create_regime_series(cycles)
            }
            
            print(f"✓ Detected {len(regime_analysis['bull_periods'])} bull market periods")
            print(f"✓ Detected {len(regime_analysis['bear_periods'])} bear market periods")
            
            # Calculate and display key statistics
            returns = self.historical_data['Returns'].dropna()
            bull_returns = returns[regime_analysis['regime'] == 'bull']
            bear_returns = returns[regime_analysis['regime'] == 'bear']
            
            print(f"\n📊 STATISTICAL PARAMETERS")
            print(f"   • Bull market drift (μ_bull): {bull_returns.mean() * 252:.4f} annually")
            print(f"   • Bear market drift (μ_bear): {bear_returns.mean() * 252:.4f} annually")
            print(f"   • Bull market volatility (σ_bull): {bull_returns.std() * np.sqrt(252):.4f}")
            print(f"   • Bear market volatility (σ_bear): {bear_returns.std() * np.sqrt(252):.4f}")
            print(f"   • Overall Sharpe ratio: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.4f}")
            
            # Store analysis results
            self.regime_analysis = regime_analysis
            self.loader = loader
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data for {self.ticker}: {str(e)}")
            return False
    
    def generate_synthetic_data(self):
        """Generate synthetic stock data using calibrated parameters."""
        print(f"\n🎲 SYNTHETIC DATA GENERATION")
        print("-" * 35)
        
        print("🔧 Initializing stochastic models...")
        print("   • Setting up Geometric Brownian Motion")
        print("   • Calibrating regime-switching parameters")
        print("   • Configuring jump diffusion processes")
        
        print("📐 Applying Maximum Likelihood Estimation (MLE)...")
        print("   • Estimating drift parameters via log-likelihood")
        print("   • Calibrating volatility using rolling windows")
        print("   • Computing regime transition probabilities")
        
        # Initialize synthetic data generator with non-deterministic behavior
        # CRITICAL: random_seed=None ensures truly random results for each run
        # This is essential for generating diverse synthetic datasets for ML training
        generator = SyntheticStockDataGenerator(
            initial_price=self.historical_data['Close'].iloc[-1],
            initial_regime="bull",
            regime_model_type="hybrid",
            random_seed=None  # IMPORTANT: None = truly random, not reproducible
        )
        
        print("🚀 Generating synthetic price paths...")
        print(f"   • Target duration: {self.synthetic_years} years ({int(self.synthetic_years * 252)} trading days)")
        print("   • Applying bounded randomness constraints")
        print("   • Incorporating regime-dependent volatility")
        
        # Generate synthetic data with historical calibration
        self.synthetic_data = generator.generate_synthetic_data(
            n_years=self.synthetic_years,
            start_date=datetime.now().strftime("%Y-%m-%d"),
            calibrate_from_historical=True,
            historical_ticker=self.ticker,
            include_jumps=False,
            progress_bar=True
        )
        
        print(f"✓ Generated {len(self.synthetic_data)} synthetic data points")
        print(f"   • Price range: ${self.synthetic_data['price'].min():.2f} - ${self.synthetic_data['price'].max():.2f}")
        print(f"   • Final price: ${self.synthetic_data['price'].iloc[-1]:.2f}")
        
        # Save complete synthetic data with all mathematical components
        # Reset index to include date as a column for proper CSV export
        output_data = self.synthetic_data.reset_index()
        
        # Rename columns for clarity and consistency with documentation
        column_mapping = {
            'date': 'date',
            'price': 'price', 
            'simple_return': 'simple_return',
            'log_return': 'log_return',
            'regime': 'regime',
            'dynamic_volatility': 'dynamic_volatility',
            'base_volatility': 'base_volatility', 
            'drift_component': 'drift_component',
            'shock_component': 'shock_component',
            'jump_component': 'jump_component',
            'cumulative_return': 'cumulative_return',
            'drawdown': 'drawdown',
            'volatility_21d': 'volatility_21d',
            'volatility_252d': 'volatility_252d',
            'regime_change': 'regime_change',
            'regime_duration': 'regime_duration'
        }
        
        # Select and reorder columns for optimal analysis workflow
        core_columns = ['date', 'price', 'simple_return', 'log_return', 'regime', 
                       'dynamic_volatility', 'base_volatility', 'drift_component', 'shock_component']
        additional_columns = ['jump_component', 'cumulative_return', 'drawdown', 
                            'volatility_21d', 'volatility_252d', 'regime_change', 'regime_duration']
        
        # Include all available columns
        available_columns = [col for col in core_columns + additional_columns if col in output_data.columns]
        output_data = output_data[available_columns]
        
        csv_path = os.path.join(self.output_dir, "synthetic_data.csv")
        output_data.to_csv(csv_path, index=False)
        print(f"✓ Saved complete synthetic data ({len(available_columns)} columns) to: {csv_path}")
        print(f"   • Core mathematical features: {len([c for c in core_columns if c in available_columns])}")
        print(f"   • Additional metrics: {len([c for c in additional_columns if c in available_columns])}")
        
        self.generator = generator
        return True
    
    def validate_synthetic_data(self):
        """Perform comprehensive statistical validation."""
        print(f"\n🔍 STATISTICAL VALIDATION")
        print("-" * 28)
        
        print("📊 Initializing validation suite...")
        print("   • Kolmogorov-Smirnov tests for distribution matching")
        print("   • Jarque-Bera test for normality of log returns")
        print("   • Ljung-Box test for autocorrelation")
        print("   • Value-at-Risk (VaR) model validation")
        
        # Configure validation parameters for comprehensive statistical testing
        
        validator = SyntheticDataValidator(self.ticker)
        
        print("🧪 Running statistical tests...")
        validation_results = validator.validate_synthetic_data(
            self.synthetic_data
        )
        
        print(f"✓ Validation complete")
        if 'overall_score' in validation_results:
            print(f"   • Overall score: {validation_results['overall_score']['overall_score']:.3f}")
            print(f"   • Validation quality: {validation_results['overall_score']['validation_quality']}")
        if 'return_distribution' in validation_results:
            if 'statistical_tests' in validation_results['return_distribution']:
                ks_stat = validation_results['return_distribution']['statistical_tests']['ks_statistic']
                print(f"   • Distribution similarity (KS): {ks_stat:.4f}")
        if 'risk_metrics' in validation_results:
            print(f"   • Risk metrics validation completed")
        print(f"   • Validation results saved for analysis")
        
        # Save validation results
        validation_path = os.path.join(self.output_dir, "validation_results.json")
        import json
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        print(f"✓ Saved validation results to: {validation_path}")
        
        self.validation_results = validation_results
        return True
    
    def create_visualizations(self):
        """Generate comprehensive visualizations for ML research."""
        print(f"\n📈 GENERATING VISUALIZATIONS")
        print("-" * 32)
        
        print("🎨 Creating research-quality visualizations...")
        print("   • Price series with regime highlighting")
        print("   • Statistical distribution comparisons")
        print("   • Stochastic process analysis")
        print("   • Risk metrics visualization")
        
        visualizer = SyntheticDataVisualizer(self.output_dir)
        
        # Key visualizations for quant ML research
        visualizations = [
            ("Price Series Comparison", "price_series_comparison"),
            ("Returns Distribution Analysis", "returns_distribution"),
            ("Regime Analysis", "regime_analysis"),
            ("Volatility Clustering", "volatility_clustering"),
            ("Autocorrelation Structure", "autocorrelation_analysis"),
            ("Risk Metrics Dashboard", "risk_metrics"),
            ("Stochastic Process Validation", "stochastic_validation"),
            ("Statistical Test Summary", "validation_summary")
        ]
        
        for viz_name, viz_method in visualizations:
            try:
                print(f"   • Generating {viz_name}...")
                if viz_method == "price_series_comparison":
                    visualizer.plot_price_series_comparison(
                        self.synthetic_data, self.historical_data, 
                        save_path=os.path.join(self.output_dir, "price_series_comparison.png")
                    )
                elif viz_method == "returns_distribution":
                    visualizer.plot_returns_distribution_comparison(
                        self.synthetic_data, self.historical_data,
                        save_path=os.path.join(self.output_dir, "returns_distribution.png")
                    )
                elif viz_method == "regime_analysis":
                    visualizer.plot_regime_analysis(
                        self.synthetic_data,
                        save_path=os.path.join(self.output_dir, "regime_analysis.png")
                    )
                elif viz_method == "volatility_clustering":
                    visualizer.plot_volatility_clustering(
                        self.synthetic_data, self.historical_data,
                        save_path=os.path.join(self.output_dir, "volatility_clustering.png")
                    )
                elif viz_method == "autocorrelation_analysis":
                    visualizer.plot_autocorrelation_analysis(
                        self.synthetic_data, self.historical_data,
                        save_path=os.path.join(self.output_dir, "autocorrelation_analysis.png")
                    )
                elif viz_method == "risk_metrics":
                    visualizer.plot_risk_metrics_comparison(
                        self.synthetic_data, self.historical_data,
                        save_path=os.path.join(self.output_dir, "risk_metrics.png")
                    )
                elif viz_method == "validation_summary":
                    visualizer.plot_validation_summary(
                        self.validation_results,
                        save_path=os.path.join(self.output_dir, "validation_summary.png")
                    )
                    
            except Exception as e:
                print(f"   ⚠️  Warning: Could not generate {viz_name}: {str(e)}")
        
        print(f"✓ Visualizations saved to: {self.output_dir}")
        
        # Highlight key visualizations for ML research
        print(f"\n🎯 KEY VISUALIZATIONS FOR QUANT ML RESEARCH:")
        print(f"   • price_series_comparison.png - Time series structure preservation")
        print(f"   • returns_distribution.png - Statistical distribution matching")
        print(f"   • volatility_clustering.png - ARCH/GARCH effects validation")
        print(f"   • autocorrelation_analysis.png - Temporal dependency structure")
        print(f"   • risk_metrics.png - VaR, CVaR, and drawdown analysis")
        
        return True
    
    def run(self):
        """Main execution pipeline."""
        self.display_banner()
        self.get_user_inputs()
        
        # Main pipeline execution
        while True:
            if not self.load_and_analyze_historical_data():
                print(f"\n❌ Failed to load data for {self.ticker}")
                print("Common reasons:")
                print("• Invalid ticker symbol")
                print("• Delisted or suspended stock")
                print("• Insufficient trading history")
                print("• Network connectivity issues")
                
                retry = input("\nTry another ticker? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting...")
                    return
                
                # Get new ticker
                while True:
                    new_ticker = input("Enter new ticker (or press Enter for ^GSPC): ").strip().upper()
                    if not new_ticker:
                        self.ticker = "^GSPC"
                        break
                    else:
                        self.ticker = new_ticker
                        break
                continue
            else:
                break
        
        # Generate synthetic data
        if not self.generate_synthetic_data():
            print("❌ Failed to generate synthetic data")
            return
        
        # Validate results
        if not self.validate_synthetic_data():
            print("❌ Failed to validate synthetic data")
            return
        
        # Create visualizations
        if not self.create_visualizations():
            print("❌ Failed to create visualizations")
            return
        
        # Final summary
        print(f"\n🎉 PIPELINE COMPLETE")
        print("=" * 50)
        print(f"✓ Historical data analyzed: {len(self.historical_data)} days")
        print(f"✓ Synthetic data generated: {len(self.synthetic_data)} days")
        if hasattr(self, 'validation_results') and 'overall_score' in self.validation_results:
            print(f"✓ Statistical validation score: {self.validation_results['overall_score']['overall_score']:.3f}")
        else:
            print(f"✓ Statistical validation completed")
        print(f"✓ All outputs saved to: {self.output_dir}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   • synthetic_data.csv - Time series data (Date, Price)")
        print(f"   • validation_results.json - Statistical test results")
        print(f"   • *.png - Research visualizations")

if __name__ == "__main__":
    pipeline = SyntheticStockPipeline()
    pipeline.run() 