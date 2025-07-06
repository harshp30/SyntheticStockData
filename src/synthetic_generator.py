"""
Synthetic Stock Data Generator
Main orchestrator that combines regime switching with stochastic processes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

from config import SIMULATION_CONFIG
from regime_model import create_regime_model
from stochastic_process import GeometricBrownianMotion, StochasticProcessFactory
from data_loader import HistoricalDataLoader

class SyntheticStockDataGenerator:
    """Main class for generating synthetic stock price data"""
    
    def __init__(self, 
                 initial_price: float = 100.0,
                 initial_regime: str = "bull",
                 regime_model_type: str = "hybrid",
                 random_seed: Optional[int] = None):
        
        self.initial_price = initial_price
        self.initial_regime = initial_regime
        self.regime_model_type = regime_model_type
        self.random_seed = random_seed
        
        # Initialize models
        self.regime_model = create_regime_model(
            regime_model_type, initial_regime, random_seed
        )
        self.stochastic_process = StochasticProcessFactory.create_gbm(
            initial_price, random_seed
        )
        
        # Data storage
        self.synthetic_data = None
        self.generation_metadata = None
        
    def generate_synthetic_data(self, 
                              n_years: float = 50.0,
                              start_date: str = "2000-01-01",
                              calibrate_from_historical: bool = True,
                              historical_ticker: str = "^GSPC",
                              include_jumps: bool = False,
                              progress_bar: bool = True) -> pd.DataFrame:
        """Generate synthetic stock price data"""
        
        print(f"Generating {n_years} years of synthetic stock data...")
        print(f"Initial price: ${self.initial_price:.2f}")
        print(f"Initial regime: {self.initial_regime}")
        print(f"Regime model: {self.regime_model_type}")
        
        # Calibrate parameters if requested
        if calibrate_from_historical:
            print("Calibrating parameters from historical data...")
            self._calibrate_parameters(historical_ticker)
            
        # Calculate number of steps (252 trading days per year)
        n_steps = int(n_years * 252)
        
        # Generate date range
        start_datetime = pd.to_datetime(start_date)
        date_range = pd.date_range(
            start=start_datetime, 
            periods=n_steps, 
            freq='B'  # Business days
        )
        
        # Initialize storage
        results = []
        
        # Progress bar
        pbar = tqdm(total=n_steps, desc="Generating data") if progress_bar else None
        
        # Generate step by step
        for i in range(n_steps):
            # Update regime
            regime_info = self.regime_model.update()
            
            # Generate price step
            price_info = self.stochastic_process.simulate_step(
                drift=regime_info['drift'],
                volatility=regime_info['volatility'],
                include_jumps=include_jumps,
                current_regime=regime_info['regime']
            )
            
            # Combine information
            step_data = {
                'date': date_range[i],
                'price': price_info['price'],
                'simple_return': price_info['simple_return'],
                'log_return': price_info['log_return'],
                'regime': regime_info['regime'],
                'regime_duration': regime_info['regime_duration'],
                'drift': regime_info['drift'],
                'volatility': regime_info['volatility'],
                'base_volatility': price_info.get('base_volatility', regime_info['volatility']),
                'dynamic_volatility': price_info.get('dynamic_volatility', regime_info['volatility']),
                'drift_component': price_info.get('drift_component', 0),
                'shock_component': price_info.get('shock_component', 0),
                'jump_component': price_info.get('jump_component', 0),
                'step': i + 1
            }
            
            results.append(step_data)
            
            if pbar:
                pbar.update(1)
                
        if pbar:
            pbar.close()
            
        # Create DataFrame
        self.synthetic_data = pd.DataFrame(results)
        self.synthetic_data.set_index('date', inplace=True)
        
        # Calculate additional metrics
        self._calculate_additional_metrics()
        
        # Store metadata
        self._store_generation_metadata(n_years, start_date, calibrate_from_historical)
        
        print(f"Generated {len(self.synthetic_data)} days of synthetic data")
        print(f"Final price: ${self.synthetic_data['price'].iloc[-1]:.2f}")
        print(f"Total return: {(self.synthetic_data['price'].iloc[-1] / self.initial_price - 1):.2%}")
        
        return self.synthetic_data
    
    def _calibrate_parameters(self, ticker: str):
        """Calibrate model parameters from historical data"""
        
        # Load historical data
        from data_loader import HistoricalDataLoader
        
        loader = HistoricalDataLoader(ticker)
        loader.fetch_data()
        loader.identify_cycles()
        loader.calculate_statistics()
        
        # Get calibrated parameters  
        try:
            params = loader.calibrate_model_parameters()
            print("Parameters calibrated from historical data")
        except:
            print("Using default parameters (historical calibration failed)")
            # Continue with default parameters from SIMULATION_CONFIG
        
    def _calculate_additional_metrics(self):
        """Calculate additional metrics for the synthetic data"""
        
        # Price-based metrics
        self.synthetic_data['cumulative_return'] = (
            self.synthetic_data['price'] / self.initial_price - 1
        )
        
        # Drawdown calculation
        self.synthetic_data['peak'] = self.synthetic_data['price'].expanding().max()
        self.synthetic_data['drawdown'] = (
            self.synthetic_data['price'] / self.synthetic_data['peak'] - 1
        )
        
        # Rolling volatility
        self.synthetic_data['volatility_21d'] = (
            self.synthetic_data['simple_return'].rolling(21).std() * np.sqrt(252)
        )
        self.synthetic_data['volatility_252d'] = (
            self.synthetic_data['simple_return'].rolling(252).std() * np.sqrt(252)
        )
        
        # Regime statistics
        self.synthetic_data['regime_change'] = (
            self.synthetic_data['regime'] != self.synthetic_data['regime'].shift(1)
        )
        
    def _store_generation_metadata(self, n_years: float, start_date: str, calibrated: bool):
        """Store metadata about the generation process"""
        
        regime_stats = self.regime_model.get_regime_statistics()
        process_stats = self.stochastic_process.get_statistics()
        
        self.generation_metadata = {
            'generation_params': {
                'n_years': n_years,
                'start_date': start_date,
                'initial_price': self.initial_price,
                'initial_regime': self.initial_regime,
                'regime_model_type': self.regime_model_type,
                'random_seed': self.random_seed,
                'calibrated_from_historical': calibrated
            },
            'regime_statistics': regime_stats,
            'process_statistics': process_stats,
            'data_statistics': self._calculate_data_statistics()
        }
        
    def _calculate_data_statistics(self) -> Dict:
        """Calculate statistics of the generated data"""
        
        if self.synthetic_data is None:
            return {}
            
        returns = self.synthetic_data['simple_return'].dropna()
        
        # Basic statistics
        stats = {
            'total_days': len(self.synthetic_data),
            'total_years': len(self.synthetic_data) / 252,
            'final_price': self.synthetic_data['price'].iloc[-1],
            'total_return': (self.synthetic_data['price'].iloc[-1] / self.initial_price) - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self.synthetic_data['drawdown'].min(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        # Regime breakdown
        regime_breakdown = self.synthetic_data['regime'].value_counts()
        stats['regime_breakdown'] = {
            'bull_days': regime_breakdown.get('bull', 0),
            'bear_days': regime_breakdown.get('bear', 0),
            'bull_percentage': regime_breakdown.get('bull', 0) / len(self.synthetic_data),
            'bear_percentage': regime_breakdown.get('bear', 0) / len(self.synthetic_data)
        }
        
        # Regime transitions
        regime_changes = self.synthetic_data['regime_change'].sum()
        stats['regime_transitions'] = regime_changes
        
        return stats
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of regime periods"""
        
        if self.synthetic_data is None:
            raise ValueError("No synthetic data generated yet")
            
        # Identify regime periods
        regime_periods = []
        current_regime = self.synthetic_data['regime'].iloc[0]
        start_idx = 0
        
        for i, regime in enumerate(self.synthetic_data['regime']):
            if regime != current_regime:
                # End of current regime
                regime_periods.append({
                    'regime': current_regime,
                    'start_date': self.synthetic_data.index[start_idx],
                    'end_date': self.synthetic_data.index[i-1],
                    'duration_days': i - start_idx,
                    'start_price': self.synthetic_data['price'].iloc[start_idx],
                    'end_price': self.synthetic_data['price'].iloc[i-1],
                    'return': (self.synthetic_data['price'].iloc[i-1] / 
                              self.synthetic_data['price'].iloc[start_idx]) - 1
                })
                
                # Start new regime
                current_regime = regime
                start_idx = i
                
        # Add final regime
        regime_periods.append({
            'regime': current_regime,
            'start_date': self.synthetic_data.index[start_idx],
            'end_date': self.synthetic_data.index[-1],
            'duration_days': len(self.synthetic_data) - start_idx,
            'start_price': self.synthetic_data['price'].iloc[start_idx],
            'end_price': self.synthetic_data['price'].iloc[-1],
            'return': (self.synthetic_data['price'].iloc[-1] / 
                      self.synthetic_data['price'].iloc[start_idx]) - 1
        })
        
        return pd.DataFrame(regime_periods)
    
    def save_data(self, filename: str, format: str = "csv"):
        """Save synthetic data to file"""
        
        if self.synthetic_data is None:
            raise ValueError("No synthetic data to save")
            
        if format.lower() == "csv":
            self.synthetic_data.to_csv(filename)
        elif format.lower() == "parquet":
            self.synthetic_data.to_parquet(filename)
        elif format.lower() == "hdf5":
            self.synthetic_data.to_hdf(filename, key='synthetic_data')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Synthetic data saved to {filename}")
    
    def load_data(self, filename: str, format: str = "csv"):
        """Load synthetic data from file"""
        
        if format.lower() == "csv":
            self.synthetic_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        elif format.lower() == "parquet":
            self.synthetic_data = pd.read_parquet(filename)
        elif format.lower() == "hdf5":
            self.synthetic_data = pd.read_hdf(filename, key='synthetic_data')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Synthetic data loaded from {filename}")
        
    def print_summary(self):
        """Print a summary of the generated data"""
        
        if self.generation_metadata is None:
            print("No generation metadata available")
            return
            
        meta = self.generation_metadata
        
        print("\n" + "="*60)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("="*60)
        
        # Generation parameters
        params = meta['generation_params']
        print(f"\nGeneration Parameters:")
        print(f"  Years simulated: {params['n_years']}")
        print(f"  Start date: {params['start_date']}")
        print(f"  Initial price: ${params['initial_price']:.2f}")
        print(f"  Initial regime: {params['initial_regime']}")
        print(f"  Regime model: {params['regime_model_type']}")
        print(f"  Random seed: {params['random_seed']}")
        print(f"  Calibrated from historical: {params['calibrated_from_historical']}")
        
        # Data statistics
        stats = meta['data_statistics']
        print(f"\nData Statistics:")
        print(f"  Total days: {stats['total_days']:,}")
        print(f"  Total years: {stats['total_years']:.1f}")
        print(f"  Final price: ${stats['final_price']:.2f}")
        print(f"  Total return: {stats['total_return']:.2%}")
        print(f"  Annual return: {stats['annual_return']:.2%}")
        print(f"  Annual volatility: {stats['annual_volatility']:.2%}")
        print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {stats['max_drawdown']:.2%}")
        
        # Regime breakdown
        regime = stats['regime_breakdown']
        print(f"\nRegime Breakdown:")
        print(f"  Bull market days: {regime['bull_days']:,} ({regime['bull_percentage']:.1%})")
        print(f"  Bear market days: {regime['bear_days']:,} ({regime['bear_percentage']:.1%})")
        print(f"  Total regime transitions: {stats['regime_transitions']}")
        
        # Regime statistics
        if meta['regime_statistics']:
            print(f"\nRegime Statistics:")
            for regime_type, regime_stats in meta['regime_statistics'].items():
                print(f"  {regime_type.title()} markets:")
                print(f"    Count: {regime_stats['count']}")
                print(f"    Avg duration: {regime_stats['avg_duration']:.1f} days")
                print(f"    Duration range: {regime_stats['min_duration']:.0f} - {regime_stats['max_duration']:.0f} days")
                
        print("="*60)

def quick_generate(n_years: float = 10.0, 
                  initial_price: float = 100.0,
                  random_seed: int = 42,
                  calibrate: bool = True) -> pd.DataFrame:
    """Quick function to generate synthetic data with default parameters"""
    
    generator = SyntheticStockDataGenerator(
        initial_price=initial_price,
        random_seed=random_seed
    )
    
    return generator.generate_synthetic_data(
        n_years=n_years,
        calibrate_from_historical=calibrate
    )

if __name__ == "__main__":
    # Example usage
    print("Testing Synthetic Data Generator...")
    
    # Create generator
    generator = SyntheticStockDataGenerator(
        initial_price=100.0,
        initial_regime="bull",
        regime_model_type="hybrid",
        random_seed=42
    )
    
    # Generate 5 years of data
    synthetic_data = generator.generate_synthetic_data(
        n_years=5.0,
        calibrate_from_historical=True
    )
    
    # Print summary
    generator.print_summary()
    
    # Show sample data
    print("\nSample Data (first 10 rows):")
    print(synthetic_data.head(10).round(4))
    
    # Get regime summary
    regime_summary = generator.get_regime_summary()
    print(f"\nRegime Summary:")
    print(regime_summary.round(4))
    
    print("\nTesting completed successfully!") 