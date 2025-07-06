# Synthetic Stock Data Generator

A stochastic stock data generator implementing regime-switching geometric Brownian motion with volatility clustering and financial stylized facts.

## Mathematical Models

### Core Stochastic Process
```
dS_t = μ(R_t)S_t dt + σ(R_t, t)S_t dW_t + J_t dN_t
```

Where:
- S_t: Stock price at time t
- R_t: Market regime (bull/bear)
- μ(R_t): Regime-dependent drift
- σ(R_t, t): Time-varying volatility
- W_t: Brownian motion
- J_t dN_t: Jump diffusion

### Key Features

**Regime Switching**: Two-state Markov chain modeling bull/bear markets
- Bull: μ = 15% annually, σ = 16% annually
- Bear: μ = -25% annually, σ = 24% annually
- Transition probabilities calibrated from historical S&P 500 data

**GARCH(1,1) Volatility Clustering**:
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```
- ω = 0.0001 (base volatility)
- α = 0.05 (short-term memory)
- β = 0.90 (long-term persistence)

**Leverage Effect**: Negative correlation between returns and future volatility
- Captures asymmetric volatility response to market moves

**Jump Diffusion**: Extreme event modeling
- Higher jump frequency during bear markets
- Downward bias in jump magnitudes

**Fat-Tailed Distributions**: Student's t-distribution with df=4.0
- Captures heavy tails observed in real market data

## Directory Structure

```
SyntheticStockData/
├── src/
│   ├── main.py                   # Main application
│   ├── config.py                 # Configuration parameters
│   ├── stochastic_process.py     # Core mathematical models
│   ├── regime_model.py           # Markov regime switching
│   ├── synthetic_generator.py    # Data generation pipeline
│   ├── data_loader.py           # Historical data processing
│   ├── validation.py            # Statistical validation
│   └── visualization.py         # Plotting and analysis
├── data/                        # Generated datasets
├── svm_trading_strategy_demo.py # ML strategy demonstration
└── README.md
```

## Usage

### Quick Start
```bash
cd src
python main.py
```

### ML Strategy Demo
```bash
python svm_trading_strategy_demo.py
```

### Programmatic Usage
```python
from synthetic_generator import SyntheticStockDataGenerator

# Create generator
generator = SyntheticStockDataGenerator(
    initial_price=100,
    random_seed=42
)

# Generate 5 years of data
data = generator.generate_synthetic_data(
    n_years=5.0,
    calibrate_from_historical=True,
    include_jumps=True
)

# Access generated data
prices = data['price']
returns = data['simple_return']
regimes = data['regime']
volatilities = data['dynamic_volatility']
```

## Output Data Structure

Generated CSV files contain 16 columns with comprehensive financial data:

**Core Features**:
- date, price, simple_return, log_return
- regime (bull/bear), dynamic_volatility, base_volatility
- drift_component, shock_component, jump_component

**Advanced Features**:
- cumulative_return, drawdown
- volatility_21d, volatility_252d
- regime_change, regime_duration

## ML Strategy Results

The SVM trading strategy demo shows measurable improvements with synthetic data augmentation:

**Historical Data Only**:
- Accuracy: 52.4%
- Sharpe Ratio: 0.350

**Historical + Synthetic Data**:
- Accuracy: 53.0% (+1.2% improvement)
- Sharpe Ratio: 0.420 (+20.2% improvement)
- Training Data: 2x increase

## Technical Implementation

**Validation**: Statistical tests including ARCH, Ljung-Box, and Kolmogorov-Smirnov
**Stylized Facts**: Heavy tails, volatility clustering, leverage effect, regime persistence
**Calibration**: Parameters estimated from historical S&P 500 data using MLE
**Stability**: Numerical bounds and safety checks throughout the pipeline