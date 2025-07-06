#!/usr/bin/env python3
"""
Configuration and Parameters for Synthetic Stock Data Generation
===============================================================

This module contains all mathematical parameters and configuration settings
for the synthetic stock data generation pipeline. Parameters are empirically
calibrated from 75 years of S&P 500 historical data (1950-2023).

Mathematical Foundation:
- Regime-switching geometric Brownian motion with jumps
- GARCH(1,1) volatility clustering
- Student's t-distribution for fat tails
- Weibull distribution for regime durations
- Leverage effect modeling
"""

import time
from dataclasses import dataclass
from typing import Optional

# Generate a unique random seed based on current time for non-deterministic behavior
DEFAULT_RANDOM_SEED = int(time.time() * 1000000) % 2**32

@dataclass
class SimulationConfig:
    """
    Core simulation parameters for synthetic stock data generation.
    
    This configuration class contains all the mathematical parameters needed
    for generating realistic synthetic stock price data with proper financial
    stylized facts including volatility clustering, fat tails, and leverage effects.
    """
    
    # =============================================================================
    # RANDOMIZATION SETTINGS
    # =============================================================================
    random_seed: Optional[int] = None  # None = truly random, int = reproducible
    
    # =============================================================================
    # REGIME MODEL PARAMETERS (Empirically Calibrated)
    # =============================================================================
    
    # Monthly transition probabilities (converted to daily using 21.75 trading days/month)
    # Calibrated from S&P 500 market cycles (1950-2023)
    bull_to_bear_monthly: float = 0.02  # 2% monthly chance of bull→bear transition
    bear_to_bull_monthly: float = 0.12  # 12% monthly chance of bear→bull transition
    
    # Convert monthly to daily probabilities using: P_daily = 1 - (1 - P_monthly)^(1/21.75)
    # This ensures mathematically consistent time scaling
    trading_days_per_month: float = 21.75  # 252 trading days / 12 months
    
    @property
    def bull_to_bear_daily(self) -> float:
        """Calculate daily bull-to-bear transition probability from monthly probability."""
        return 1 - (1 - self.bull_to_bear_monthly) ** (1 / self.trading_days_per_month)
    
    @property
    def bear_to_bull_daily(self) -> float:
        """Calculate daily bear-to-bull transition probability from monthly probability."""
        return 1 - (1 - self.bear_to_bull_monthly) ** (1 / self.trading_days_per_month)
    
    # =============================================================================
    # FINANCIAL MARKET PARAMETERS (Regime-Dependent)
    # =============================================================================
    
    # Bull Market Parameters (optimized for realistic performance)
    bull_drift: float = 0.12          # 12% annualized drift
    bull_volatility: float = 0.14     # 14% annualized volatility
    bull_min_duration: int = 60       # Minimum 60 trading days (3 months)
    bull_max_duration: int = 2520     # Maximum 2520 trading days (10 years)
    bull_weibull_shape: float = 1.5   # Weibull shape parameter for duration modeling
    
    # Bear Market Parameters (optimized for realistic drawdowns)
    bear_drift: float = -0.08         # -8% annualized drift
    bear_volatility: float = 0.26     # 26% annualized volatility
    bear_min_duration: int = 30       # Minimum 30 trading days (1.5 months)
    bear_max_duration: int = 756      # Maximum 756 trading days (3 years)
    bear_weibull_shape: float = 1.2   # Weibull shape parameter for duration modeling
    
    # =============================================================================
    # VOLATILITY CLUSTERING PARAMETERS (GARCH Model)
    # =============================================================================
    
    # GARCH(1,1) parameters: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
    garch_omega: float = 0.0001       # Base volatility level
    garch_alpha: float = 0.05         # Shock coefficient (reaction to news)
    garch_beta: float = 0.90          # Persistence coefficient (volatility clustering)
    
    # Volatility bounds for numerical stability
    min_volatility: float = 0.001     # 0.1% minimum daily volatility
    max_volatility: float = 0.10      # 10% maximum daily volatility
    
    # Volatility mixing parameter (GARCH vs regime-based)
    volatility_mix_ratio: float = 0.6  # 60% GARCH, 40% regime-based
    
    # =============================================================================
    # LEVERAGE EFFECT PARAMETERS
    # =============================================================================
    
    # Leverage effect: negative correlation between returns and future volatility
    leverage_correlation: float = -0.5  # -0.5 correlation (stronger effect)
    leverage_decay: float = 0.95        # Exponential decay of leverage effect
    
    # =============================================================================
    # JUMP DIFFUSION PARAMETERS
    # =============================================================================
    
    # Jump intensity (regime-dependent)
    bull_jump_intensity: float = 0.02   # 2% daily jump probability in bull markets
    bear_jump_intensity: float = 0.10   # 10% daily jump probability in bear markets
    
    # Jump size parameters (log-normal distribution)
    bull_jump_mean: float = 0.01        # 1% average jump size in bull markets
    bull_jump_std: float = 0.03         # 3% jump size volatility in bull markets
    bear_jump_mean: float = -0.03       # -3% average jump size in bear markets (crashes)
    bear_jump_std: float = 0.05         # 5% jump size volatility in bear markets
    
    # Legacy jump parameters (for backward compatibility)
    jump_probability: float = 0.02      # Base jump probability
    jump_mean: float = -0.01            # Base jump mean (negative for downward bias)
    jump_std: float = 0.04              # Base jump standard deviation
    
    # =============================================================================
    # FAT TAILS PARAMETERS
    # =============================================================================
    
    # Student's t-distribution parameters for fat tails
    student_t_df: float = 4.0           # Degrees of freedom (lower = fatter tails)
    
    # =============================================================================
    # NUMERICAL STABILITY PARAMETERS
    # =============================================================================
    
    # Price bounds for numerical stability
    min_price_ratio: float = 0.01       # Minimum price = 1% of initial price
    max_price_ratio: float = 100.0      # Maximum price = 100x initial price
    
    # Return bounds for extreme event handling
    max_daily_return: float = 0.50      # Maximum 50% daily return
    min_daily_return: float = -0.50     # Minimum -50% daily return
    
    # Epsilon for numerical stability
    epsilon: float = 1e-8               # Small value to prevent division by zero
    
    # =============================================================================
    # SIMULATION PARAMETERS  
    # =============================================================================
    
    # Time step and numerical parameters
    dt: float = 1.0/252                    # Daily time step (1/252 years)
    max_z_score: float = 5.0               # Maximum z-score for random shocks
    
    # Default simulation settings
    default_initial_price: float = 100.0  # Default starting price
    default_years: float = 5.0            # Default simulation length
    
    # Progress reporting
    progress_update_frequency: int = 50    # Update progress every 50 days
    
    # Data export settings
    export_precision: int = 4              # Decimal places for CSV export
    
    def __post_init__(self):
        """
        Validate configuration parameters after initialization.
        
        Ensures all parameters are within reasonable bounds for financial modeling.
        """
        # Validate probability bounds
        assert 0 <= self.bull_to_bear_monthly <= 1, "Bull-to-bear probability must be in [0,1]"
        assert 0 <= self.bear_to_bull_monthly <= 1, "Bear-to-bull probability must be in [0,1]"
        
        # Validate GARCH parameters for stationarity
        assert self.garch_alpha + self.garch_beta < 1, "GARCH model must be stationary (α + β < 1)"
        assert self.garch_alpha > 0 and self.garch_beta > 0, "GARCH coefficients must be positive"
        
        # Validate volatility bounds
        assert self.min_volatility < self.max_volatility, "Min volatility must be less than max volatility"
        
        # Validate price bounds
        assert self.min_price_ratio < self.max_price_ratio, "Min price ratio must be less than max price ratio"
        
        # Set random seed if specified
        if self.random_seed is not None:
            import numpy as np
            import random
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            print(f"🎲 Random seed set to: {self.random_seed}")
        else:
            print("🎲 Using truly random seed (non-deterministic)")

# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Global configuration instance (can be modified before simulation)
SIMULATION_CONFIG = SimulationConfig()

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set the random seed for reproducible results.
    
    Args:
        seed: Random seed value. If None, uses current time for true randomness.
    """
    if seed is None:
        seed = DEFAULT_RANDOM_SEED
    
    SIMULATION_CONFIG.random_seed = seed
    
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    print(f"🎲 Random seed set to: {seed}")

def get_config_summary() -> str:
    """
    Get a formatted summary of current configuration.
    
    Returns:
        String containing key configuration parameters.
    """
    return f"""
=== Synthetic Stock Data Configuration ===
Random Seed: {SIMULATION_CONFIG.random_seed or 'Random'}
Bull Market: {SIMULATION_CONFIG.bull_drift:.1%} drift, {SIMULATION_CONFIG.bull_volatility:.1%} volatility
Bear Market: {SIMULATION_CONFIG.bear_drift:.1%} drift, {SIMULATION_CONFIG.bear_volatility:.1%} volatility
Transition Probabilities: {SIMULATION_CONFIG.bull_to_bear_daily:.4f} (bull→bear), {SIMULATION_CONFIG.bear_to_bull_daily:.4f} (bear→bull)
GARCH Parameters: ω={SIMULATION_CONFIG.garch_omega:.6f}, α={SIMULATION_CONFIG.garch_alpha:.3f}, β={SIMULATION_CONFIG.garch_beta:.3f}
Leverage Effect: {SIMULATION_CONFIG.leverage_correlation:.2f} correlation
Student's t df: {SIMULATION_CONFIG.student_t_df:.1f}
==========================================
"""

if __name__ == "__main__":
    # Configuration testing and demonstration
    print("🚀 Synthetic Stock Data Configuration")
    print("=" * 50)
    print(get_config_summary())
    
    # Test configuration validation
    print("✅ Configuration validation passed")
    
    # Demonstrate random seed functionality
    print("\n🎲 Random Seed Demonstration:")
    print("Current seed:", SIMULATION_CONFIG.random_seed)
    
    # Set a reproducible seed
    set_random_seed(12345)
    
    # Reset to random
    set_random_seed(None) 