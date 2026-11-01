# Legacy IEEE KalmanNet Baseline (KBPT)

This folder contains the original baseline implementation from the IEEE KalmanNet pairs trading literature:
- `main_ieee.py`: Original IEEE benchmark runner
- `KalmanNet_nn.py`: Original generic KalmanNet neural network architectures (Cohen & Shlezinger)
- `KalmanNet_sysmdl.py`: Linear system model wrapper
- `Pipeline_EKF.py`: Extended Kalman Filter / Neural training pipeline
- `Extended_data.py`: Synthetic trajectory generator and decimation utilities
- `Linear_KF.py`: Classical linear Kalman filter baseline
- `KalmanFilter_test.py`: KF test suite
- `Utilities.py`: Utility functions and multi-page plotting
- `positions.py`: Position sizing and thresholds

For the modern Dynamic Partial Cointegration (Dynamic-PCI) pipeline and regularized KalmanNet architecture, refer to the root directory modules (`dynamic_pci.py`, `neural_kalman_pci.py`, `regime_filter.py`, `backtest.py`).
