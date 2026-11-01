flowchart TD
    A["1. Data Ingestion (data_loader.py)"] -->|DataSet/ CSV Cache or yfinance| B["Clean, Synchronize & Base-1.0 Normalize"]
    B --> C["Split: In-Sample (Formation) & Out-of-Sample (Trading)"]
    C --> D["2. Parameter Estimation (dynamic_pci.py)"]
    D -->|MLE via L-BFGS-B on Formation Data| E["Fitted Parameters (beta0, rho, sigma_M, sigma_R, q_beta)"]
    E --> SCR["3. Statistical Screening (regime_filter.py)"]
    SCR -->|Check R2_MR >= 15%, ADF p <= 0.10, Half-Life <= 252d| DEC{"Pair Suitable?"}
    DEC -->|No: e.g. AAP-IBM| REJ["Reject Pair / Suppress Capital Allocation"]
    DEC -->|Yes: e.g. KO-PEP| FILT["4. Real-Time State Filtering"]
    FILT --> KF["Classical Kalman Filter (dynamic_pci.py)"]
    FILT --> KNET["Neural KalmanNet (neural_kalman_pci.py)"]
    KF -->|4-State Filter: [X1, beta, M, R]| OUT["Filter Trajectories (beta_t, M_t)"]
    KNET -->|Learned Gain via GRU & Sharpe Loss| OUT
    OUT --> GAT["5. Online Regime Gating (regime_filter.py)"]
    GAT -->|Rolling Half-Life & Volatility Ratio| BT["6. Out-of-Sample Backtesting (backtest.py)"]
    BT -->|Z-Score Trading + 10 bps Transaction Cost| RES["Risk & Return Metrics (Sharpe, Return, Max DD)"]
    RES --> DASH["7. Visualizations & Reports (outputs/)"]
