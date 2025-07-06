"""
Historical Data Loader and Analysis Module
Fetches real market data and analyzes bull/bear cycles for model calibration
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

class HistoricalDataLoader:
    """
    Historical Stock Market Data Loader and Analyzer
    ===============================================
    
    This class handles fetching, processing, and analyzing historical stock market data
    from Yahoo Finance. It implements advanced cycle detection algorithms to identify
    bull and bear market periods using drawdown analysis.
    
    Key Features:
    - Automatic data fetching from Yahoo Finance API
    - Bull/bear market cycle detection using drawdown thresholds
    - Statistical parameter calculation for regime modeling
    - Comprehensive data validation and preprocessing
    - Support for any ticker symbol with sufficient trading history
    
    Mathematical Methods:
    - Drawdown Analysis: DD(t) = (P(t) / Peak(t)) - 1
    - Return Calculations: Simple and log returns
    - Volatility Estimation: Rolling standard deviation with annualization
    - Regime Classification: Threshold-based cycle identification
    
    Usage:
        loader = HistoricalDataLoader("^GSPC")  # S&P 500
        data = loader.fetch_data("2020-01-01", "2023-12-31")
        cycles = loader.identify_cycles()
        stats = loader.calculate_statistics()
    """
    
    def __init__(self, ticker: str = "^GSPC"):
        """
        Initialize the historical data loader.
        
        Args:
            ticker: Stock ticker symbol (default: S&P 500)
        """
        self.ticker = ticker
        self.data = None
        self.cycles = None
        self.statistics = None
        
    def fetch_data(self, start_date: str = "1950-01-01", 
                   end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None = today)
            
        Returns:
            DataFrame with OHLCV data plus calculated metrics
        """
        # Use today's date if end_date is None
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        print(f"Fetching data for {self.ticker} from {start_date} to {end_date}...")
        
        try:
            # Download data with progress indicator
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if self.data.empty:
                raise ValueError(f"No data retrieved for {self.ticker}")
                
            # Calculate returns
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            # Calculate cumulative returns and drawdowns
            self.data['Cumulative_Returns'] = (1 + self.data['Returns']).cumprod()
            self.data['Peak'] = self.data['Close'].expanding().max()
            self.data['Drawdown'] = (self.data['Close'] / self.data['Peak']) - 1
            
            # Calculate rolling volatility
            self.data['Volatility_21d'] = self.data['Returns'].rolling(21).std() * np.sqrt(252)
            self.data['Volatility_252d'] = self.data['Returns'].rolling(252).std() * np.sqrt(252)
            
            print(f"Successfully loaded {len(self.data)} trading days of data")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
            
    def identify_cycles(self, bear_threshold: float = -0.20,
                       min_duration: int = 60) -> List[Dict]:
        """Identify bull and bear market cycles"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
            
        print("Identifying bull and bear market cycles...")
        
        cycles = []
        prices = self.data['Close'].copy()
        dates = self.data.index
        
        # Find peaks and troughs
        current_peak = prices.iloc[0]
        current_peak_date = dates[0]
        current_trough = prices.iloc[0]
        current_trough_date = dates[0]
        
        in_bear_market = False
        bear_start_date = None
        bear_start_price = None
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Update peaks
            if price > current_peak:
                current_peak = price
                current_peak_date = date
                
                # End bear market if we were in one
                if in_bear_market and (date - bear_start_date).days >= min_duration:
                    cycles.append({
                        'type': 'bear',
                        'start_date': bear_start_date,
                        'end_date': date,
                        'start_price': bear_start_price,
                        'end_price': price,
                        'duration_days': (date - bear_start_date).days,
                        'return': (price / bear_start_price) - 1,
                        'max_drawdown': (current_trough / bear_start_price) - 1
                    })
                    in_bear_market = False
            
            # Check for bear market start
            drawdown = (price / current_peak) - 1
            if drawdown <= bear_threshold and not in_bear_market:
                # Start of bear market
                in_bear_market = True
                bear_start_date = current_peak_date
                bear_start_price = current_peak
                current_trough = price
                current_trough_date = date
            
            # Update troughs during bear market
            if in_bear_market and price < current_trough:
                current_trough = price
                current_trough_date = date
                
        # Handle ongoing bear market at end of data
        if in_bear_market and (dates[-1] - bear_start_date).days >= min_duration:
            cycles.append({
                'type': 'bear',
                'start_date': bear_start_date,
                'end_date': dates[-1],
                'start_price': bear_start_price,
                'end_price': prices.iloc[-1],
                'duration_days': (dates[-1] - bear_start_date).days,
                'return': (prices.iloc[-1] / bear_start_price) - 1,
                'max_drawdown': (current_trough / bear_start_price) - 1
            })
            
        # Identify bull markets (periods between bear markets)
        bear_cycles = [c for c in cycles if c['type'] == 'bear']
        
        # Add bull market before first bear (if any)
        if bear_cycles:
            if bear_cycles[0]['start_date'] > dates[0]:
                first_bear_start_idx = self.data.index.get_loc(bear_cycles[0]['start_date'])
                bull_return = (bear_cycles[0]['start_price'] / prices.iloc[0]) - 1
                
                cycles.append({
                    'type': 'bull',
                    'start_date': dates[0],
                    'end_date': bear_cycles[0]['start_date'],
                    'start_price': prices.iloc[0],
                    'end_price': bear_cycles[0]['start_price'],
                    'duration_days': (bear_cycles[0]['start_date'] - dates[0]).days,
                    'return': bull_return,
                    'max_drawdown': self.data['Drawdown'].loc[dates[0]:bear_cycles[0]['start_date']].min()
                })
        
        # Add bull markets between bear markets
        for i in range(len(bear_cycles) - 1):
            bull_start_date = bear_cycles[i]['end_date']
            bull_end_date = bear_cycles[i+1]['start_date']
            bull_start_price = bear_cycles[i]['end_price']
            bull_end_price = bear_cycles[i+1]['start_price']
            
            bull_return = (bull_end_price / bull_start_price) - 1
            duration = (bull_end_date - bull_start_date).days
            
            if duration >= min_duration:
                cycles.append({
                    'type': 'bull',
                    'start_date': bull_start_date,
                    'end_date': bull_end_date,
                    'start_price': bull_start_price,
                    'end_price': bull_end_price,
                    'duration_days': duration,
                    'return': bull_return,
                    'max_drawdown': self.data['Drawdown'].loc[bull_start_date:bull_end_date].min()
                })
                
        # Add bull market after last bear (if any)
        if bear_cycles and bear_cycles[-1]['end_date'] < dates[-1]:
            last_bear_end_price = bear_cycles[-1]['end_price']
            bull_return = (prices.iloc[-1] / last_bear_end_price) - 1
            duration = (dates[-1] - bear_cycles[-1]['end_date']).days
            
            if duration >= min_duration:
                cycles.append({
                    'type': 'bull',
                    'start_date': bear_cycles[-1]['end_date'],
                    'end_date': dates[-1],
                    'start_price': last_bear_end_price,
                    'end_price': prices.iloc[-1],
                    'duration_days': duration,
                    'return': bull_return,
                    'max_drawdown': self.data['Drawdown'].loc[bear_cycles[-1]['end_date']:].min()
                })
        
        # Sort cycles by start date
        cycles.sort(key=lambda x: x['start_date'])
        self.cycles = cycles
        
        bull_count = len([c for c in cycles if c['type'] == 'bull'])
        bear_count = len([c for c in cycles if c['type'] == 'bear'])
        print(f"Identified {bull_count} bull markets and {bear_count} bear markets")
        
        return cycles
    
    def calculate_statistics(self) -> Dict:
        """Calculate statistical parameters from historical data and cycles"""
        if self.data is None or self.cycles is None:
            raise ValueError("Data and cycles must be loaded first")
            
        print("Calculating statistical parameters...")
        
        # Basic return statistics
        returns = self.data['Returns'].dropna()
        log_returns = self.data['Log_Returns'].dropna()
        
        # Overall statistics
        overall_stats = {
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': self.data['Drawdown'].min(),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }
        
        # Cycle statistics
        bull_cycles = [c for c in self.cycles if c['type'] == 'bull']
        bear_cycles = [c for c in self.cycles if c['type'] == 'bear']
        
        bull_stats = {}
        bear_stats = {}
        
        if bull_cycles:
            bull_durations = [c['duration_days'] for c in bull_cycles]
            bull_returns = [c['return'] for c in bull_cycles]
            
            bull_stats = {
                'count': len(bull_cycles),
                'avg_duration_days': np.mean(bull_durations),
                'median_duration_days': np.median(bull_durations),
                'avg_return': np.mean(bull_returns),
                'median_return': np.median(bull_returns),
                'avg_annual_return': np.mean([(1 + r) ** (365.25 / d) - 1 for r, d in zip(bull_returns, bull_durations)]),
                'volatility': np.std([self.data['Returns'].loc[c['start_date']:c['end_date']].std() * np.sqrt(252) 
                                    for c in bull_cycles if len(self.data.loc[c['start_date']:c['end_date']]) > 21])
            }
            
        if bear_cycles:
            bear_durations = [c['duration_days'] for c in bear_cycles]
            bear_returns = [c['return'] for c in bear_cycles]
            
            bear_stats = {
                'count': len(bear_cycles),
                'avg_duration_days': np.mean(bear_durations),
                'median_duration_days': np.median(bear_durations),
                'avg_return': np.mean(bear_returns),
                'median_return': np.median(bear_returns),
                'avg_annual_return': np.mean([(1 + r) ** (365.25 / d) - 1 for r, d in zip(bear_returns, bear_durations)]),
                'volatility': np.std([self.data['Returns'].loc[c['start_date']:c['end_date']].std() * np.sqrt(252) 
                                    for c in bear_cycles if len(self.data.loc[c['start_date']:c['end_date']]) > 21])
            }
        
        self.statistics = {
            'overall': overall_stats,
            'bull_markets': bull_stats,
            'bear_markets': bear_stats,
            'data_period': {
                'start_date': self.data.index[0],
                'end_date': self.data.index[-1],
                'total_days': len(self.data),
                'years': (self.data.index[-1] - self.data.index[0]).days / 365.25
            }
        }
        
        return self.statistics
    
    def get_regime_data(self, regime_type: str) -> pd.DataFrame:
        """Get data points that belong to a specific regime"""
        if self.data is None or self.cycles is None:
            raise ValueError("Data and cycles must be loaded first")
            
        regime_cycles = [c for c in self.cycles if c['type'] == regime_type]
        regime_data = pd.DataFrame()
        
        for cycle in regime_cycles:
            cycle_data = self.data.loc[cycle['start_date']:cycle['end_date']].copy()
            cycle_data['regime'] = regime_type
            cycle_data['cycle_id'] = f"{regime_type}_{cycle['start_date'].strftime('%Y%m%d')}"
            regime_data = pd.concat([regime_data, cycle_data])
            
        return regime_data
    
    def calibrate_model_parameters(self) -> Dict:
        """Calibrate model parameters based on historical analysis"""
        if self.statistics is None:
            raise ValueError("Statistics must be calculated first")
            
        print("Calibrating model parameters from historical data...")
        
        bull_stats = self.statistics['bull_markets']
        bear_stats = self.statistics['bear_markets']
        
        # Convert durations to years for annual parameters
        bull_duration_years = bull_stats['avg_duration_days'] / 365.25 if bull_stats else 3.7
        bear_duration_years = bear_stats['avg_duration_days'] / 365.25 if bear_stats else 1.0
        
        # Calculate transition probabilities based on average durations
        # P(exit) = 1 / expected_duration_in_periods
        # For monthly transitions: P_monthly = 12 / duration_years
        bull_to_bear_monthly = min(0.5, 12 / bull_duration_years) if bull_duration_years > 0 else 0.025
        bear_to_bull_monthly = min(0.5, 12 / bear_duration_years) if bear_duration_years > 0 else 0.083
        
        calibrated_params = {
            'bull_drift_annual': bull_stats.get('avg_annual_return', 0.15),
            'bull_volatility_annual': bull_stats.get('volatility', 0.16),
            'bull_avg_duration_years': bull_duration_years,
            'bear_drift_annual': bear_stats.get('avg_annual_return', -0.25),
            'bear_volatility_annual': bear_stats.get('volatility', 0.24),
            'bear_avg_duration_years': bear_duration_years,
            'bull_to_bear_prob': bull_to_bear_monthly,
            'bear_to_bull_prob': bear_to_bull_monthly,
            'overall_annual_return': self.statistics['overall']['annual_return'],
            'overall_volatility': self.statistics['overall']['annual_volatility']
        }
        
        return calibrated_params
    
    def print_summary(self):
        """Print a summary of the historical analysis"""
        if self.statistics is None:
            print("No analysis performed yet. Run the full analysis first.")
            return
            
        stats = self.statistics
        
        print("\n" + "="*60)
        print("HISTORICAL MARKET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nData Period: {stats['data_period']['start_date'].strftime('%Y-%m-%d')} to {stats['data_period']['end_date'].strftime('%Y-%m-%d')}")
        print(f"Total Years: {stats['data_period']['years']:.1f}")
        print(f"Trading Days: {stats['data_period']['total_days']:,}")
        
        print(f"\nOverall Performance:")
        print(f"  Annual Return: {stats['overall']['annual_return']:.2%}")
        print(f"  Annual Volatility: {stats['overall']['annual_volatility']:.2%}")
        print(f"  Sharpe Ratio: {stats['overall']['sharpe_ratio']:.2f}")
        print(f"  Maximum Drawdown: {stats['overall']['max_drawdown']:.2%}")
        
        if stats['bull_markets']:
            bull = stats['bull_markets']
            print(f"\nBull Markets ({bull['count']} cycles):")
            print(f"  Average Duration: {bull['avg_duration_days']:.0f} days ({bull['avg_duration_days']/365.25:.1f} years)")
            print(f"  Average Return: {bull['avg_return']:.2%}")
            print(f"  Average Annual Return: {bull['avg_annual_return']:.2%}")
            print(f"  Average Volatility: {bull.get('volatility', 0):.2%}")
            
        if stats['bear_markets']:
            bear = stats['bear_markets']
            print(f"\nBear Markets ({bear['count']} cycles):")
            print(f"  Average Duration: {bear['avg_duration_days']:.0f} days ({bear['avg_duration_days']/365.25:.1f} years)")
            print(f"  Average Return: {bear['avg_return']:.2%}")
            print(f"  Average Annual Return: {bear['avg_annual_return']:.2%}")
            print(f"  Average Volatility: {bear.get('volatility', 0):.2%}")
            
        print("="*60)

def load_and_analyze_historical_data(ticker: str = "^GSPC") -> HistoricalDataLoader:
    """Convenience function to load and analyze historical data"""
    loader = HistoricalDataLoader(ticker)
    
    # Load data
    loader.fetch_data()
    
    # Analyze cycles
    loader.identify_cycles()
    
    # Calculate statistics
    loader.calculate_statistics()
    
    # Print summary
    loader.print_summary()
    
    return loader

if __name__ == "__main__":
    # Example usage
    loader = load_and_analyze_historical_data()
    
    # Get calibrated parameters
    params = loader.calibrate_model_parameters()
    print(f"\nCalibrated Parameters:")
    for key, value in params.items():
        if 'prob' in key or 'return' in key or 'volatility' in key:
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:.2f}") 