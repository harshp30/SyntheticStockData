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
    """GBM with GARCH volatility clustering and leverage effect."""
    
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
        
        # GARCH(1,1) parameters: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        self.enable_volatility_clustering = enable_volatility_clustering
        self.garch_alpha = garch_alpha  # Short-term memory
        self.garch_beta = garch_beta    # Long-term persistence
        self.garch_omega = garch_omega  # Base volatility
        self.current_volatility = None
        
        # Leverage effect: negative returns increase volatility more
        self.leverage_correlation = leverage_correlation
        
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            # Random seed for non-deterministic behavior
            import time
            np.random.seed(int(time.time() * 1000000) % 2**32)
            
        # Simulation parameters
        self.dt = SIMULATION_CONFIG.dt
        self.max_z_score = SIMULATION_CONFIG.max_z_score
        
    def generate_random_shock(self, volatility: float, 
                            distribution: str = "student_t",
                            bounded: bool = True,
                            df: float = 4.0) -> float:
        """Generate random shock with fat tails (Student's t-distribution)."""
        
        if distribution == "normal":
            z = np.random.normal(0, 1)
        elif distribution == "student_t":
            # Student's t-distribution with fat tails, scaled to unit variance
            if df > 2:
                scale_factor = np.sqrt(df / (df - 2))
                z = np.random.standard_t(df) / scale_factor
            else:
                z = np.random.normal(0, 1)
        elif distribution == "laplace":
            # Laplace distribution (double exponential)
            z = np.random.laplace(0, 1/np.sqrt(2))
        elif distribution == "mixture":
            # Mixture of normal distributions
            if np.random.random() < 0.9:
                z = np.random.normal(0, 1)
            else:
                z = np.random.normal(0, 2.0)  # Fat tail regime
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        # Apply bounds to prevent extreme values
        if bounded:
            z = np.clip(z, -self.max_z_score, self.max_z_score)
            
        return volatility * z
    
    def _calculate_dynamic_volatility(self, base_volatility: float) -> float:
        """Calculate GARCH(1,1) volatility with leverage effect:
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1} + leverage_term
        """
        if not self.enable_volatility_clustering:
            return base_volatility
            
        # Initialize volatility on first step
        if self.current_volatility is None:
            self.current_volatility = base_volatility
            self.volatility_history.append(base_volatility)
            return base_volatility
            
        # Calculate GARCH volatility based on previous period
        if len(self.return_history) > 0:
            prev_return = self.return_history[-1]
            prev_volatility = self.volatility_history[-1]
            
            # Bound return to prevent extreme values
            prev_return = np.clip(prev_return, -0.5, 0.5)
            
            # Standardized residual for GARCH
            epsilon = 1e-8
            prev_std_residual = prev_return / (prev_volatility + epsilon)
            prev_std_residual = np.clip(prev_std_residual, -5, 5)
            
            # Leverage effect: negative returns increase volatility more
            leverage_term = 0.0
            if prev_return < 0:
                leverage_multiplier = min(abs(self.leverage_correlation), 0.3)
                leverage_term = leverage_multiplier * (prev_return**2) / (prev_volatility**2 + epsilon)
            
            # GARCH(1,1) equation with leverage effect
            volatility_squared = (self.garch_omega + 
                                self.garch_alpha * prev_std_residual**2 + 
                                self.garch_beta * prev_volatility**2 +
                                leverage_term)
            
            # Ensure positive and bounded volatility
            volatility_squared = np.clip(volatility_squared, 1e-8, 1.0)
            new_volatility = np.sqrt(volatility_squared)
            
            # Apply bounds relative to base volatility
            min_vol = max(base_volatility * 0.5, 0.005)
            max_vol = min(base_volatility * 2.0, 0.10)
            new_volatility = np.clip(new_volatility, min_vol, max_vol)
            
            # Blend with base volatility to maintain regime characteristics
            blend_factor = 0.6
            new_volatility = blend_factor * new_volatility + (1 - blend_factor) * base_volatility
            
        else:
            new_volatility = base_volatility
            
        # Final bounds check
        new_volatility = np.clip(new_volatility, 1e-6, 1.0)
        
        self.current_volatility = new_volatility
        self.volatility_history.append(new_volatility)
        
        return new_volatility
        
    def simulate_step(self, drift: float, volatility: float, 
                     include_jumps: bool = False, current_regime: str = "bull") -> Dict[str, float]:
        """Simulate one step of GBM: dS = μS dt + σS dW + jumps"""
        
        # Calculate dynamic volatility with GARCH clustering
        dynamic_volatility = self._calculate_dynamic_volatility(volatility)
        
        # Generate random shock (Student's t-distribution)
        shock = self.generate_random_shock(dynamic_volatility)
        
        # Jump diffusion component
        jump_component = 0.0
        if include_jumps:
            jump_component = self._generate_jump(current_regime)
            
        # GBM drift component with Ito correction
        drift_component = (drift - 0.5 * dynamic_volatility**2) * self.dt
        
        # Total log return
        log_return = drift_component + shock + jump_component
        
        # Bound log return to prevent extreme price movements
        log_return = np.clip(log_return, -0.15, 0.15)
        
        # Update price (exponential ensures positive prices)
        new_price = self.current_price * np.exp(log_return)
        
        # Price bounds to prevent extremes
        min_price = self.initial_price * 0.10
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
        """Generate jump component with regime-dependent parameters and downward bias."""
        config = SIMULATION_CONFIG
        
        # Regime-dependent jump parameters
        if current_regime == "bear":
            jump_prob = config.jump_probability * 2.0  # More frequent jumps in bear markets
            jump_mean = config.jump_mean * 1.2  # Stronger negative bias
            jump_std = config.jump_std * 0.8
        else:
            jump_prob = config.jump_probability * 0.5  # Fewer jumps in bull markets
            jump_mean = config.jump_mean * 0.5
            jump_std = config.jump_std * 0.6
        
        # Volatility adjustment: higher volatility = more jumps
        if len(self.volatility_history) > 0:
            current_vol = self.volatility_history[-1]
            vol_multiplier = current_vol / 0.16  # Normalize to 16% annual
            jump_prob *= vol_multiplier
            
        # Generate jump if occurs
        if np.random.random() < jump_prob:
            # Mixture of normal and extreme jumps
            if np.random.random() < 0.7:
                jump_size = np.random.normal(jump_mean, jump_std)
            else:
                # Extreme jump (fat tail)
                extreme_mean = jump_mean * 2.0
                extreme_std = jump_std * 1.5
                jump_size = np.random.normal(extreme_mean, extreme_std)
            
            # Downward bias: negative jumps are more extreme (crashes)
            if jump_size < 0:
                jump_size *= 1.3
                
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
    """Factory for creating different GBM configurations."""
    
    @staticmethod
    def create_gbm(initial_price: float = 100.0,
                  random_seed: Optional[int] = None,
                  enable_volatility_clustering: bool = True,
                  leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create GBM with GARCH volatility clustering."""
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
        """Create GBM with jump diffusion."""
        return GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
    
    @staticmethod
    def create_bounded_process(initial_price: float = 100.0,
                             max_daily_move: float = 0.15,
                             random_seed: Optional[int] = None,
                             enable_volatility_clustering: bool = True,
                             leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create GBM with bounded daily moves."""
        gbm = GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
        # Convert max daily move to z-score bound
        gbm.max_z_score = max_daily_move / 0.01
        return gbm
    
    @staticmethod
    def create_fat_tail_process(initial_price: float = 100.0,
                              random_seed: Optional[int] = None,
                              df: float = 4.0,
                              enable_volatility_clustering: bool = True,
                              leverage_correlation: float = -0.5) -> GeometricBrownianMotion:
        """Create GBM with fat-tailed returns (Student's t)."""
        gbm = GeometricBrownianMotion(
            initial_price=initial_price,
            random_seed=random_seed,
            enable_volatility_clustering=enable_volatility_clustering,
            leverage_correlation=leverage_correlation
        )
        gbm.df = df  # Degrees of freedom for t-distribution
        return gbm

if __name__ == "__main__":
    # Simple test
    gbm = GeometricBrownianMotion(100.0, random_seed=42)
    
    # Simulate 1 year
    for i in range(252):
        result = gbm.simulate_step(0.10/252, 0.20/np.sqrt(252))
        
    stats = gbm.get_statistics()
    print(f"Final price: ${gbm.current_price:.2f}")
    print(f"Annual return: {stats['annual_return']:.2%}")
    print(f"Annual volatility: {stats['annual_volatility']:.2%}") 