# options-pricing-tool
Simple option pricing tool
Overview
This tool prices European Call and Put options and computes all five major Greeks (risk sensitivities), then renders a 6-panel matplotlib dashboard to visualize how the option behaves across different market conditions.
Built as a self-contained Python script — no external data feed required.

Features

Black-Scholes pricing for Call and Put options
Full Greeks calculation:
GreekMeasuresDelta (Δ)Price sensitivity to stock movementGamma (Γ)Rate of change of DeltaTheta (Θ)Time decay per calendar dayVega (ν)Sensitivity to volatility changesRho (ρ)Sensitivity to interest rate changes

6-panel visual dashboard:

Option price vs. stock price (Call & Put)
Payoff diagram at expiry
Delta curve
Gamma curve
Theta decay over time
Vega vs. volatility


Greeks summary table with Call vs. Put comparison


Getting Started
Prerequisites
bashpip install numpy scipy matplotlib
Run
bashpython options_pricing_tool.py
Customise Parameters
Edit the five variables at the bottom of the script:
pythonS     = 150.0   # Current stock price ($)
K     = 155.0   # Strike price ($)
T     = 0.25    # Time to expiry in years (0.25 = 3 months)
r     = 0.05    # Risk-free interest rate (0.05 = 5%)
sigma = 0.25    # Implied volatility (0.25 = 25%)

📊 Example Output
With default parameters (S=150, K=155, T=3mo, r=5%, σ=25%):
  CALL  →  $6.1138
    Delta    +0.46024
    Gamma    +0.02117
    Theta    -0.04940
    Vega     +0.29772
    Rho      +0.15731

  PUT   →  $9.1883
    Delta    -0.53976
    Gamma    +0.02117
    Theta    -0.02843
    Vega     +0.29772
    Rho      -0.22538

🧠 How It Works
The Black-Scholes formula prices a European option under these assumptions:

The underlying follows a log-normal random walk
No dividends, no early exercise
Constant volatility and risk-free rate​

🗂️ Project Structure
options-pricing-tool/
├── options_pricing_tool.py   # Main script
├── options_dashboard.png     # Sample dashboard output
└── README.md

🛠️ Built With

NumPy — numerical computation
SciPy — normal distribution functions
Matplotlib — visualizations


📄 License
MIT License — free to use, modify, and distribute.
