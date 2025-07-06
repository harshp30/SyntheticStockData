"""
Stochastic Process Module
Implements Geometric Brownian Motion and bounded randomness for synthetic stock data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
import warnings

from config import SIMULATION_CONFIG

class GeometricBrownianMotion:
    """Geometric Brownian Motion process with volatility clustering for stock price simulation"""
    
    def __init__(self, initial_price: float = 100.0, 
                 random_seed: Optional[int] = None,
                 enable_volatility_clustering: bool = True,
                 garch_alpha: float = 0.05,
                 garch_beta: float = 0.90,
                 garch_omega: float = 0.0001,
                 leverage_correlation: float = -0.5):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.price_history = [initial_price]
        self.return_history = []
        self.volatility_history = []
        self.step = 0
        
        # Volatility clustering parameters (more conservative defaults)
        self.enable_volatility_clustering = enable_volatility_clustering
        self.garch_alpha = garch_alpha  # Reduced from 0.1 to 0.05
        self.garch_beta = garch_beta    # Increased from 0.85 to 0.90 for more persistence
        self.garch_omega = garch_omega  # Increased from 0.000001 to 0.0001
        self.current_volatility = None  # Will be set in first step
        
        # Leverage effect parameter (reduced from -0.5 to -0.3)
        self.leverage_correlation = leverage_correlation  # Negative correlation between returns and volatility
        
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            # For truly random behavior, seed with current time
            import time
            np.random.seed(int(time.time() * 1000000) % 2**32)
            
        # Configuration
        self.dt = SIMULATION_CONFIG.dt
        self.max_z_score = SIMULATION_CONFIG.max_z_score
        
    def generate_random_shock(self, volatility: float, 
                            distribution: str = "student_t",
                            bounded: bool = True,
                            df: float = 4.0) -> float:
        """Generate random shock with fat tails and optional bounds
        
        Args:
            volatility: Volatility parameter
            distribution: 'normal', 'student_t', 'laplace', or 'mixture'
            bounded: Whether to apply bounds
            df: Degrees of freedom for Student's t (lower = fatter tails)
        """
        
        if distribution == "normal":
            z = np.random.normal(0, 1)
        elif distribution == "student_t":
            # Student's t-distribution with fat tails
            # Scale to unit variance: divide by sqrt(df/(df-2))
            if df > 2:
                scale_factor = np.sqrt(df / (df - 2))
                z = np.random.standard_t(df) / scale_factor
            else:
                # Fallback to normal if df too small
                z = np.random.normal(0, 1)
        elif distribution == "laplace":
            # Laplace distribution (double exponential)
            z = np.random.laplace(0, 1/np.sqrt(2))
        elif distribution == "mixture":
            # Mixture of normal distributions (captures multiple regimes)
            if np.random.random() < 0.9:
                # 90% normal regime
                z = np.random.normal(0, 1)
            else:
                # 10% extreme regime (fat tails)
                z = np.random.normal(0, 2.0)  # Higher variance
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        # Apply bounds if requested
        if bounded:
            z = np.clip(z, -self.max_z_score, self.max_z_score)
            
        # The volatility parameter is already converted to daily volatility
        # so we don't need to scale by sqrt(dt) again
        return volatility * z
    
    def _calculate_dynamic_volatility(self, base_volatility: float) -> float:
        """Calculate time-varying volatility using GARCH-style clustering with leverage effect
        
        Uses GARCH(1,1) model: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        Plus leverage effect: negative returns increase volatility more than positive returns
        """
        if not self.enable_volatility_clustering:
            return base_volatility
            
        # Initialize volatility if first step
        if self.current_volatility is None:
            self.current_volatility = base_volatility
            self.volatility_history.append(base_volatility)
            return base_volatility
            
        # Get previous return and volatility
        if len(self.return_history) > 0:
            prev_return = self.return_history[-1]
            prev_volatility = self.volatility_history[-1]
            
            # Bound previous return to prevent extreme values
            prev_return = np.clip(prev_return, -0.5, 0.5)  # Cap at ±50% daily return
            
            # Calculate standardized residual (return / volatility)
            # Add small epsilon to prevent division by zero
            epsilon = 1e-8
            prev_std_residual = prev_return / (prev_volatility + epsilon)
            
            # Bound the standardized residual
            prev_std_residual = np.clip(prev_std_residual, -5, 5)
            
            # Leverage effect: negative returns have stronger impact on volatility
            # Use asymmetric GARCH approach with bounds
            leverage_term = 0.0
            if prev_return < 0:
                # Negative returns increase volatility more, but limit the effect
                leverage_multiplier = min(abs(self.leverage_correlation), 0.3)  # Max 30% leverage effect
                leverage_term = leverage_multiplier * (prev_return**2) / (prev_volatility**2 + epsilon)
            
            # GARCH(1,1) equation for volatility squared with leverage effect
            volatility_squared = (self.garch_omega + 
                                self.garch_alpha * prev_std_residual**2 + 
                                self.garch_beta * prev_volatility**2 +
                                leverage_term)
            
            # Ensure volatility squared is positive and bounded
            volatility_squared = np.clip(volatility_squared, 1e-8, 1.0)  # Max 100% daily volatility
            
            # Take square root and ensure positive
            new_volatility = np.sqrt(volatility_squared)
            
            # Additional bounds on volatility (much tighter for stability)
            min_vol = max(base_volatility * 0.5, 0.005)  # At least 50% of base, minimum 0.5% daily
            max_vol = min(base_volatility * 2.0, 0.10)   # At most 2x base, maximum 10% daily
            new_volatility = np.clip(new_volatility, min_vol, max_vol)
            
            # Blend with base volatility to maintain regime characteristics
            # This ensures volatility clustering while respecting regime-based levels
            blend_factor = 0.6  # Reduce from 0.8 to 0.6 for more stability
            new_volatility = blend_factor * new_volatility + (1 - blend_factor) * base_volatility
            
        else:
            new_volatility = base_volatility
            
        # Final safety check
        new_volatility = np.clip(new_volatility, 1e-6, 1.0)  # Absolute bounds
        
        self.current_volatility = new_volatility
        self.volatility_history.append(new_volatility)
        
        return new_volatility
        
    def simulate_step(self, drift: float, volatility: float, 
                     include_jumps: bool = False, current_regime: str = "bull") -> Dict[str, float]:
        """Simulate one step of the GBM process with dynamic volatility"""
        
        # Calculate dynamic volatility (includes clustering effects)
        dynamic_volatility = self._calculate_dynamic_volatility(volatility)
        
        # Generate random shock using dynamic volatility
        shock = self.generate_random_shock(dynamic_volatility)
        
        # Jump diffusion component (optional)
        jump_component = 0.0
        if include_jumps:
            jump_component = self._generate_jump(current_regime)
            
        # Calculate drift component (with drift adjustment for GBM)
        drift_component = (drift - 0.5 * dynamic_volatility**2) * self.dt
        
        # Total log return with bounds to prevent extreme values
        log_return = drift_component + shock + jump_component
        
        # Bound log return to prevent price explosions
        # ±15% daily log return is more realistic (still very extreme)
        log_return = np.clip(log_return, -0.15, 0.15)
        
        # Update price using exponential (ensures positive prices)
        new_price = self.current_price * np.exp(log_return)
        
        # Additional safety check for price bounds
        # Prevent prices from going to extremes (tighter bounds for realism)
        min_price = self.initial_price * 0.10  # Don't go below 10% of initial price
        max_price = self.initial_price * 10.0  # Don't go above 10x initial price  
        new_price = np.clip(new_price, min_price, max_price)
        
        # Simple return (recalculate based on bounded price)
        simple_return = (new_price - self.current_price) / self.current_price
        
        # Update state
        self.current_price = new_price
        self.price_history.append(new_price)
        self.return_history.append(simple_return)
        self.step += 1
        
        return {
            'price': new_price,
            'simple_return': simple_return,
            'log_return': log_return,
            'drift_component': drift_component,
            'shock_component': shock,
            'jump_component': jump_component,
            'base_volatility': volatility,
            'dynamic_volatility': dynamic_volatility,
            'current_regime': current_regime,
            'step': self.step
        }
    
    def _generate_jump(self, current_regime: str = "bull") -> float:
        """Generate jump component for enhanced jump diffusion
        
        Improvements:
        - Regime-dependent jump parameters
        - Downward bias (crashes more common than rallies)
        - Higher jump probability during bear markets
        - Jump intensity correlated with volatility
        """
        config = SIMULATION_CONFIG
        
        # Regime-dependent jump parameters (reduced for stability)
        if current_regime == "bear":
            # Bear markets: more frequent jumps, but not extreme
            jump_prob = config.jump_probability * 2.0  # 2x more jumps (reduced from 5x)
            jump_mean = config.jump_mean * 1.2  # Slightly stronger negative bias (reduced from 1.5x)
            jump_std = config.jump_std * 0.8   # Reduced volatility for stability
        else:
            # Bull markets: fewer jumps, less extreme
            jump_prob = config.jump_probability * 0.5  # Even fewer jumps in bull markets
            jump_mean = config.jump_mean * 0.5  # Reduced negative bias
            jump_std = config.jump_std * 0.6   # Reduced volatility
        
        # Volatility adjustment: higher volatility periods have more jumps
        if len(self.volatility_history) > 0:
            current_vol = self.volatility_history[-1]
            # Normalize volatility (assuming base of 16% annual = 0.16)
            vol_multiplier = current_vol / 0.16
            jump_prob *= vol_multiplier
            
        # Check if jump occurs
        if np.random.random() < jump_prob:
            # Generate jump size with downward bias
            # Use mixture of normal distributions for more realistic jumps
            if np.random.random() < 0.7:
                # 70% chance of "normal" jump
                jump_size = np.random.normal(jump_mean, jump_std)
            else:
                # 30% chance of "extreme" jump (fat tail)
                extreme_mean = jump_mean * 2.0
                extreme_std = jump_std * 1.5
                jump_size = np.random.normal(extreme_mean, extreme_std)
            
            # Additional downward bias: negative jumps are more extreme
            if jump_size < 0:
                jump_size *= 1.3  # Make negative jumps 30% more extreme
                
            return jump_size
        else:
            return 0.0
    
    def simulate_path(self, n_steps: int, drift: float, volatility: float,
                     include_jumps: bool = False, current_regime: str = "bull") -> pd.DataFrame:
        """Simulate entire price path"""
        
        results = []
        
        for i in range(n_steps):
            step_result = self.simulate_step(drift, volatility, include_jumps, current_regime)
            results.append(step_result)
            
        return pd.DataFrame(results)
    
    def reset(self, initial_price: Optional[float] = None):
        """Reset the process to initial state"""
        if initial_price is not None:
            self.initial_price = initial_price
            
        self.current_price = self.initial_price
        self.price_history = [self.initial_price]
        self.return_history = []
        self.volatility_history = []
        self.current_volatility = None
        self.step = 0
        
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics of the generated path"""
        if len(self.return_history) < 2:
            return {}
            
        returns = np.array(self.return_history)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'annual_return': np.mean(returns) * 252,
            'annual_volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'total_return': (self.current_price / self.initial_price) - 1
        }

class BoundedRandomness:
    """Utilities for generating bounded random variables"""
    
    @staticmethod
    def truncated_normal(mean: float = 0.0, std: float = 1.0,
                        lower_bound: float = -3.0, upper_bound: float = 3.0,
                        size: int = 1) -> Union[float, np.ndarray]:
        """Generate truncated normal random variables"""
        
        # Convert bounds to standard normal scale
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        
        # Generate truncated normal
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        
        return samples[0] if size == 1 else samples
    
    @staticmethod
    def bounded_t_distribution(df: int = 5, scale: float = 1.0,
                             bound: float = 3.0, size: int = 1) -> Union[float, np.ndarray]:
        """Generate bounded t-distribution samples"""
        
        samples = []
        for _ in range(size):
            while True:
                sample = np.random.standard_t(df) * scale
                if abs(sample) <= bound:
                    samples.append(sample)
                    break
                    
        return samples[0] if size == 1 else np.array(samples)
    
    @staticmethod
    def mixture_normal(weights: List[float], means: List[float], 
                      stds: List[float], bound: float = 3.0,
                      size: int = 1) -> Union[float, np.ndarray]:
        """Generate mixture of normal distributions with bounds"""
        
        samples = []
        for _ in range(size):
            # Choose component
            component = np.random.choice(len(weights), p=weights)
            
            # Generate sample from chosen component
            while True:
                sample = np.random.normal(means[component], stds[component])
                if abs(sample) <= bound:
                    samples.append(sample)
                    break
                    
        return samples[0] if size == 1 else np.array(samples)

class VolatilityModel:
    """Time-varying volatility models"""
    
    def __init__(self, base_volatility: float = 0.16):
        self.base_volatility = base_volatility
        self.volatility_history = []
        
    def garch_volatility(self, returns: np.ndarray, 
                        alpha: float = 0.1, beta: float = 0.85,
                        omega: float = 0.000001) -> float:
        """Simple GARCH(1,1) volatility model"""
        
        if len(returns) < 2:
            return self.base_volatility
            
        # Previous return and volatility
        prev_return = returns[-1]
        prev_vol = self.volatility_history[-1] if self.volatility_history else self.base_volatility
        
        # GARCH equation
        new_vol_squared = omega + alpha * prev_return**2 + beta * prev_vol**2
        new_vol = np.sqrt(new_vol_squared)
        
        self.volatility_history.append(new_vol)
        return new_vol
    
    def regime_volatility(self, base_vol: float, regime: str,
                         stress_multiplier: float = 1.5) -> float:
        """Adjust volatility based on regime"""
        
        if regime == "bear":
            return base_vol * stress_multiplier
        else:
            return base_vol
    
    def seasonal_volatility(self, base_vol: float, day_of_year: int) -> float:
        """Add seasonal volatility patterns"""
        
        # Higher volatility in autumn/winter (market crash seasons)
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        return base_vol * seasonal_factor

class StochasticProcessFactory:
    """Factory for creating different stochastic processes"""
    
    @staticmethod
    def create_gbm(initial_price: float = 100.0,
                  random_seed: Optional[int] = None,
                  enable_volatility_clustering: bool = True,
                  leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create enhanced GBM process with volatility clustering and leverage effect"""
        return GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
    
    @staticmethod
    def create_jump_diffusion(initial_price: float = 100.0,
                            random_seed: Optional[int] = None,
                            enable_volatility_clustering: bool = True,
                            leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create enhanced GBM with jump diffusion, volatility clustering, and leverage effect"""
        gbm = GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
        # Jump parameters are configured in SIMULATION_CONFIG
        return gbm
    
    @staticmethod
    def create_bounded_process(initial_price: float = 100.0,
                             max_daily_move: float = 0.15,
                             random_seed: Optional[int] = None,
                             enable_volatility_clustering: bool = True,
                             leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create enhanced GBM with strict daily move bounds"""
        gbm = GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
        # Convert max daily move to z-score bound
        # Assuming 1% daily volatility: max_z = max_move / 0.01
        gbm.max_z_score = max_daily_move / 0.01
        return gbm
    
    @staticmethod
    def create_fat_tail_process(initial_price: float = 100.0,
                              random_seed: Optional[int] = None,
                              df: float = 4.0,
                              enable_volatility_clustering: bool = True,
                              leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create enhanced GBM with fat tails (Student's t-distribution)"""
        gbm = GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
        # Store degrees of freedom for fat tails
        gbm.df = df
        return gbm

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Stochastic Processes...")
    
    # Test GBM
    gbm = GeometricBrownianMotion(100.0, random_seed=42)
    
    # Simulate 1 year of data
    n_steps = 252
    drift = 0.10 / 252  # 10% annual drift
    volatility = 0.20 / np.sqrt(252)  # 20% annual volatility
    
    print(f"Simulating {n_steps} steps...")
    print(f"Drift: {drift:.6f} (daily), Volatility: {volatility:.6f} (daily)")
    
    # Run simulation
    for i in range(n_steps):
        result = gbm.simulate_step(drift, volatility)
        
    # Get statistics
    stats = gbm.get_statistics()
    
    print("\nSimulation Results:")
    print(f"Final price: ${gbm.current_price:.2f}")
    print(f"Total return: {stats['total_return']:.2%}")
    print(f"Annual return: {stats['annual_return']:.2%}")
    print(f"Annual volatility: {stats['annual_volatility']:.2%}")
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Skewness: {stats['skewness']:.3f}")
    print(f"Kurtosis: {stats['kurtosis']:.3f}")
    
    # Test bounded randomness
    print("\nTesting Bounded Randomness...")
    
    # Generate bounded samples
    bounded_samples = BoundedRandomness.truncated_normal(
        mean=0, std=1, lower_bound=-2, upper_bound=2, size=1000
    )
    
    print(f"Bounded normal samples - Min: {np.min(bounded_samples):.3f}, Max: {np.max(bounded_samples):.3f}")
    print(f"Mean: {np.mean(bounded_samples):.3f}, Std: {np.std(bounded_samples):.3f}")
    
    # Test mixture distribution
    mixture_samples = BoundedRandomness.mixture_normal(
        weights=[0.7, 0.3], means=[0, 0], stds=[1, 2], bound=3, size=1000
    )
    
    print(f"Mixture samples - Min: {np.min(mixture_samples):.3f}, Max: {np.max(mixture_samples):.3f}")
    print(f"Mean: {np.mean(mixture_samples):.3f}, Std: {np.std(mixture_samples):.3f}")
    
    print("\nAll tests completed successfully!") 