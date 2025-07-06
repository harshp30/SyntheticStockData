"""
Market Regime Switching Model
Implements bull/bear market regime transitions using Markov chains or duration sampling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random
from scipy.special import gamma

from config import SIMULATION_CONFIG

class RegimeType(Enum):
    """Enumeration for market regime types"""
    BULL = "bull"
    BEAR = "bear"

@dataclass
class RegimeState:
    """Current state of the market regime"""
    regime: RegimeType
    start_date: int  # Step number when regime started
    duration: int    # Steps elapsed in current regime
    remaining_duration: Optional[int] = None  # For duration-based model

class MarkovRegimeModel:
    """Markov chain-based regime switching model"""
    
    def __init__(self, initial_regime: str = "bull", random_seed: Optional[int] = None):
        self.current_regime = RegimeType(initial_regime)
        self.transition_probs = {
            'bull_to_bear': SIMULATION_CONFIG.bull_to_bear_daily,
            'bear_to_bull': SIMULATION_CONFIG.bear_to_bull_daily
        }
        self.regime_history = []
        self.step = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        else:
            # For truly random behavior, seed with current time
            import time
            seed = int(time.time() * 1000000) % 2**32
            np.random.seed(seed)
            random.seed(seed)
            
        # Initialize state
        self.state = RegimeState(
            regime=self.current_regime,
            start_date=0,
            duration=0
        )
        
    def should_transition(self) -> bool:
        """Determine if regime should transition based on Markov probabilities"""
        if self.current_regime == RegimeType.BULL:
            prob = self.transition_probs["bull_to_bear"]
        else:
            prob = self.transition_probs["bear_to_bull"]
            
        return np.random.random() < prob
    
    def transition_regime(self) -> RegimeType:
        """Transition to the opposite regime"""
        if self.current_regime == RegimeType.BULL:
            new_regime = RegimeType.BEAR
        else:
            new_regime = RegimeType.BULL
            
        # Record the completed regime
        self.regime_history.append({
            'regime': self.current_regime.value,
            'start_step': self.state.start_date,
            'end_step': self.step,
            'duration': self.state.duration
        })
        
        # Update state
        self.current_regime = new_regime
        self.state = RegimeState(
            regime=new_regime,
            start_date=self.step,
            duration=0
        )
        
        return new_regime
    
    def update(self) -> Dict[str, Union[str, float, int]]:
        """Update regime for one time step"""
        # Check for transition
        if self.should_transition():
            self.transition_regime()
            
        # Update duration
        self.state.duration += 1
        self.step += 1
        
        # Get current regime parameters
        if self.current_regime == RegimeType.BULL:
            params = {
                'drift': SIMULATION_CONFIG.bull_drift,
                'volatility': SIMULATION_CONFIG.bull_volatility
            }
        else:
            params = {
                'drift': SIMULATION_CONFIG.bear_drift,
                'volatility': SIMULATION_CONFIG.bear_volatility
            }
        
        return {
            'regime': self.current_regime.value,
            'step': self.step,
            'regime_duration': self.state.duration,
            'drift': params['drift'],
            'volatility': params['volatility']
        }
    
    def get_regime_statistics(self) -> Dict[str, Dict]:
        """Get statistics about regime durations and transitions"""
        if not self.regime_history:
            return {}
            
        df = pd.DataFrame(self.regime_history)
        
        stats = {}
        for regime in ['bull', 'bear']:
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                stats[regime] = {
                    'count': len(regime_data),
                    'avg_duration': regime_data['duration'].mean(),
                    'median_duration': regime_data['duration'].median(),
                    'min_duration': regime_data['duration'].min(),
                    'max_duration': regime_data['duration'].max(),
                    'std_duration': regime_data['duration'].std()
                }
                
        return stats

class DurationBasedRegimeModel:
    """Duration-based regime switching model that samples regime lengths"""
    
    def __init__(self, initial_regime: str = "bull", random_seed: Optional[int] = None):
        self.current_regime = RegimeType(initial_regime)
        self.regime_history = []
        self.step = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        else:
            # For truly random behavior, seed with current time
            import time
            seed = int(time.time() * 1000000) % 2**32
            np.random.seed(seed)
            random.seed(seed)
            
        # Sample initial regime duration
        initial_duration = self._sample_regime_duration(self.current_regime)
        
        self.state = RegimeState(
            regime=self.current_regime,
            start_date=0,
            duration=0,
            remaining_duration=initial_duration
        )
        
    def _sample_regime_duration(self, regime: RegimeType) -> int:
        """Sample duration for a regime based on historical distributions
        
        Uses Weibull distribution which better captures the empirical distribution
        of market regime durations compared to exponential distribution.
        Shape parameter k=1.5 gives realistic right-skewed distribution.
        """
        # Get average regime duration from configuration
        if regime == RegimeType.BULL:
            avg_duration = 756  # ~3 years for bull markets
        else:
            avg_duration = 252  # ~1 year for bear markets
        
        # Use Weibull distribution for regime durations
        # Shape parameter k controls the distribution shape:
        # k=1.5 gives realistic right-skewed distribution for financial regimes
        # k=1 would be exponential (too simple)
        # k=2 would be Rayleigh (less realistic for regimes)
        if regime == RegimeType.BULL:
            shape_param = 1.5  # Bull markets have moderate variability
        else:
            shape_param = 1.2  # Bear markets are more variable (lower shape)
        
        # Scale parameter to match desired average duration
        # For Weibull: mean = scale * Gamma(1 + 1/shape)
        scale_param = avg_duration / gamma(1 + 1/shape_param)
        
        # Sample from Weibull distribution
        duration = np.random.weibull(shape_param) * scale_param
        
        # Ensure minimum duration (avoid very short regimes)
        if regime == RegimeType.BEAR:
            min_duration = 30  # 1 month minimum for bear markets
            max_duration = 756  # 3 years maximum (very long bear markets are rare)
        else:
            min_duration = 60   # 2 months minimum for bull markets
            max_duration = 2520  # 10 years maximum (very long bull markets are rare)
        
        duration = np.clip(duration, min_duration, max_duration)
        
        return int(duration)
    
    def should_transition(self) -> bool:
        """Check if current regime duration is completed"""
        return self.state.remaining_duration <= 0
    
    def transition_regime(self) -> RegimeType:
        """Transition to the opposite regime with new sampled duration"""
        # Record completed regime
        self.regime_history.append({
            'regime': self.current_regime.value,
            'start_step': self.state.start_date,
            'end_step': self.step,
            'duration': self.state.duration,
            'planned_duration': self.state.duration + abs(self.state.remaining_duration)
        })
        
        # Switch regime
        if self.current_regime == RegimeType.BULL:
            new_regime = RegimeType.BEAR
        else:
            new_regime = RegimeType.BULL
            
        # Sample new duration
        new_duration = self._sample_regime_duration(new_regime)
        
        # Update state
        self.current_regime = new_regime
        self.state = RegimeState(
            regime=new_regime,
            start_date=self.step,
            duration=0,
            remaining_duration=new_duration
        )
        
        return new_regime
    
    def update(self) -> Dict[str, Union[str, float, int]]:
        """Update regime for one time step"""
        # Check for transition
        if self.should_transition():
            self.transition_regime()
            
        # Update durations
        self.state.duration += 1
        self.state.remaining_duration -= 1
        self.step += 1
        
        # Get current regime parameters
        if self.current_regime == RegimeType.BULL:
            params = {
                'drift': SIMULATION_CONFIG.bull_drift,
                'volatility': SIMULATION_CONFIG.bull_volatility
            }
        else:
            params = {
                'drift': SIMULATION_CONFIG.bear_drift,
                'volatility': SIMULATION_CONFIG.bear_volatility
            }
        
        return {
            'regime': self.current_regime.value,
            'step': self.step,
            'regime_duration': self.state.duration,
            'remaining_duration': self.state.remaining_duration,
            'drift': params['drift'],
            'volatility': params['volatility']
        }
    
    def get_regime_statistics(self) -> Dict[str, Dict]:
        """Get statistics about regime durations"""
        if not self.regime_history:
            return {}
            
        df = pd.DataFrame(self.regime_history)
        
        stats = {}
        for regime in ['bull', 'bear']:
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                stats[regime] = {
                    'count': len(regime_data),
                    'avg_duration': regime_data['duration'].mean(),
                    'median_duration': regime_data['duration'].median(),
                    'min_duration': regime_data['duration'].min(),
                    'max_duration': regime_data['duration'].max(),
                    'std_duration': regime_data['duration'].std(),
                    'avg_planned_duration': regime_data['planned_duration'].mean()
                }
                
        return stats

class HybridRegimeModel:
    """Hybrid model that combines Markov transitions with duration constraints"""
    
    def __init__(self, initial_regime: str = "bull", 
                 min_duration_months: Dict[str, int] = None,
                 random_seed: Optional[int] = None):
        
        self.current_regime = RegimeType(initial_regime)
        self.regime_history = []
        self.step = 0
        
        # Minimum durations to prevent unrealistic short regimes
        self.min_durations = min_duration_months or {
            'bull': 6 * 21,   # 6 months * ~21 trading days
            'bear': 2 * 21    # 2 months * ~21 trading days
        }
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        else:
            # For truly random behavior, seed with current time
            import time
            seed = int(time.time() * 1000000) % 2**32
            np.random.seed(seed)
            random.seed(seed)
            
        # Get transition probabilities
        self.transition_probs = {
            'bull_to_bear': SIMULATION_CONFIG.bull_to_bear_daily,
            'bear_to_bull': SIMULATION_CONFIG.bear_to_bull_daily
        }
        
        self.state = RegimeState(
            regime=self.current_regime,
            start_date=0,
            duration=0
        )
        
    def should_transition(self) -> bool:
        """Check transition based on Markov probability and minimum duration"""
        # Enforce minimum duration
        min_duration = self.min_durations[self.current_regime.value]
        if self.state.duration < min_duration:
            return False
            
        # Use Markov probability
        if self.current_regime == RegimeType.BULL:
            prob = self.transition_probs["bull_to_bear"]
        else:
            prob = self.transition_probs["bear_to_bull"]
            
        return np.random.random() < prob
    
    def transition_regime(self) -> RegimeType:
        """Transition to opposite regime"""
        # Record completed regime
        self.regime_history.append({
            'regime': self.current_regime.value,
            'start_step': self.state.start_date,
            'end_step': self.step,
            'duration': self.state.duration
        })
        
        # Switch regime
        if self.current_regime == RegimeType.BULL:
            new_regime = RegimeType.BEAR
        else:
            new_regime = RegimeType.BULL
            
        self.current_regime = new_regime
        self.state = RegimeState(
            regime=new_regime,
            start_date=self.step,
            duration=0
        )
        
        return new_regime
    
    def update(self) -> Dict[str, Union[str, float, int]]:
        """Update regime for one time step"""
        # Check for transition
        if self.should_transition():
            self.transition_regime()
            
        # Update duration
        self.state.duration += 1
        self.step += 1
        
        # Get current regime parameters
        if self.current_regime == RegimeType.BULL:
            params = {
                'drift': SIMULATION_CONFIG.bull_drift,
                'volatility': SIMULATION_CONFIG.bull_volatility
            }
        else:
            params = {
                'drift': SIMULATION_CONFIG.bear_drift,
                'volatility': SIMULATION_CONFIG.bear_volatility
            }
        
        return {
            'regime': self.current_regime.value,
            'step': self.step,
            'regime_duration': self.state.duration,
            'drift': params['drift'],
            'volatility': params['volatility']
        }
    
    def get_regime_statistics(self) -> Dict[str, Dict]:
        """Get statistics about regime durations and transitions"""
        if not self.regime_history:
            return {}
            
        df = pd.DataFrame(self.regime_history)
        
        stats = {}
        for regime in ['bull', 'bear']:
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                stats[regime] = {
                    'count': len(regime_data),
                    'avg_duration': regime_data['duration'].mean(),
                    'median_duration': regime_data['duration'].median(),
                    'min_duration': regime_data['duration'].min(),
                    'max_duration': regime_data['duration'].max(),
                    'std_duration': regime_data['duration'].std()
                }
                
        return stats

def create_regime_model(model_type: str = "hybrid", 
                       initial_regime: str = "bull",
                       random_seed: Optional[int] = None) -> Union[MarkovRegimeModel, DurationBasedRegimeModel, HybridRegimeModel]:
    """Factory function to create regime models"""
    
    if model_type.lower() == "markov":
        return MarkovRegimeModel(initial_regime, random_seed)
    elif model_type.lower() == "duration":
        return DurationBasedRegimeModel(initial_regime, random_seed)
    elif model_type.lower() == "hybrid":
        return HybridRegimeModel(initial_regime, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'markov', 'duration', 'hybrid'")

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Regime Models...")
    
    # Test different models
    models = {
        "Markov": create_regime_model("markov", random_seed=42),
        "Duration": create_regime_model("duration", random_seed=42),
        "Hybrid": create_regime_model("hybrid", random_seed=42)
    }
    
    # Simulate for 10 years
    n_steps = 10 * 252  # 10 years of trading days
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        print("-" * 40)
        
        regime_sequence = []
        for step in range(n_steps):
            result = model.update()
            regime_sequence.append(result['regime'])
            
        # Analyze results
        stats = model.get_regime_statistics()
        
        bull_count = len([r for r in regime_sequence if r == 'bull'])
        bear_count = len([r for r in regime_sequence if r == 'bear'])
        
        print(f"Bull days: {bull_count} ({bull_count/n_steps:.1%})")
        print(f"Bear days: {bear_count} ({bear_count/n_steps:.1%})")
        
        if stats:
            for regime, regime_stats in stats.items():
                print(f"{regime.title()} markets:")
                print(f"  Count: {regime_stats['count']}")
                print(f"  Avg duration: {regime_stats['avg_duration']:.1f} days ({regime_stats['avg_duration']/252:.1f} years)") 