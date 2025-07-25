import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple


@njit
def _ewm_std(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Exponential Weighted Moving Std (EWMA Std)."""
    n = len(data)
    alpha = 2 / (period + 1)
    mean = np.empty(n, dtype=np.float64)
    var = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    mean[0] = data[0]
    var[0] = 0.0
    std[0] = 0.0
    for i in range(1, n):
        mean[i] = alpha * data[i] + (1 - alpha) * mean[i - 1]
        var[i] = (1 - alpha) * (var[i - 1] + alpha * (data[i] - mean[i - 1]) ** 2)
        std[i] = np.sqrt(var[i])
    return std

@njit
def _sma_std(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute simple rolling standard deviation (SMA Std)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        window = data[i - period:i]
        result[i] = np.std(window)
    return result

def rolling_std(data: NDArray[np.float64], period: int, method: str = "EWMA") -> NDArray[np.float64]:
    """
    General rolling standard deviation wrapper.

    Args:
        data: Price series
        period: Window size
        method: 'EWMA' or 'SMA'

    Returns:
        Standard deviation series.
    """
    method = method.upper()
    if method == "EWMA":
        return _ewm_std(data, period)
    elif method == "SMA":
        return _sma_std(data, period)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'EWMA' or 'SMA'.")

@njit
def _ema(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Exponential Moving Average."""
    n = len(data)
    alpha = 2 / (period + 1)
    result = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

@njit
def _sma(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple Moving Average (SMA)."""
    n = len(data)
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma

def moving_average(
    data: NDArray[np.float64],
    period: int,
    method: str = "EMA"
) -> NDArray[np.float64]:
    """Moving Average (MA): Simple, EMA, SMA."""
    if method.lower() == "ema":
        return _ema(data, period)
    elif method.lower() == "sma":
        return _sma(data, period)
    else:
        raise ValueError("Invalid MA type. Choose 'EMA', 'SMA'.")

@njit
def _macd(
    data: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute MACD indicator.

    Returns:
        - macd_line: EMA(fast) - EMA(slow)
        - signal_line: EMA(macd_line, signal_period)
        - histogram: macd_line - signal_line
    """
    fast_ema = _ema(data, fast_period)
    slow_ema = _ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _calculate_bollinger_bands(
    close: NDArray[np.float64],
    period: int,
    std_multiplier: float = 2.0,
    ma_method: str = "EMA",
    std_type: str = "EWMA"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bollinger Bands (upper, mid, lower)."""
    std = rolling_std(close, period, std_type)
    mid = moving_average(close, period, ma_method)
    upper = mid + std_multiplier * std
    lower = mid - std_multiplier * std
    return upper, mid, lower

@njit
def _rsi(
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Relative Strength Index (RSI)."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    rs = np.zeros(n, dtype=np.float64)
    rsi = np.full(n, np.nan, dtype=np.float64)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        if avg_loss[i] == 0:
            rs[i] = np.inf
            rsi[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    return rsi

@njit
def _average_true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Average True Range (ATR)."""
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = np.abs(high[i] - close[i - 1])
        lc = np.abs(low[i] - close[i - 1])
        tr = max(hl, hc, lc)
        if i == period:
            total = 0.0
            for j in range(1, period + 1):
                hl = high[j] - low[j]
                hc = np.abs(high[j] - close[j - 1])
                lc = np.abs(low[j] - close[j - 1])
                total += max(hl, hc, lc)
            atr[i] = total / period
        elif i > period:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr

def _trend_detect(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
    threshold: float = 0.4
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Detect trend based on ATR/std ratio."""
    atr_vals = _average_true_range(high, low, close, period)
    std_vals = moving_average(close, period, "EMA")
    ratio = atr_vals / std_vals
    n = len(ratio)
    mean_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        mean_ratio[i] = np.mean(ratio[i - 3:i])
    trending = np.where(mean_ratio <= threshold, 1, 0)
    return mean_ratio, trending

@njit
def _calculate_volatility_normalize(
    data: NDArray[np.float64],
    timeperiod: int = 14
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize volatility from log returns."""
    n = len(data)
    log_returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        log_returns[i] = np.log(data[i] / data[i - 1])
    vol = np.zeros(n, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        r_slice = log_returns[i - timeperiod + 1:i + 1]
        std = np.std(r_slice)
        vol[i] = std * np.sqrt(timeperiod)
    recent_vol = vol[-timeperiod:]
    vmin = np.min(recent_vol)
    vmax = np.max(recent_vol)
    normalized_vol = np.zeros_like(recent_vol)
    if vmax > vmin:
        for i in range(timeperiod):
            normalized_vol[i] = (recent_vol[i] - vmin) / (vmax - vmin)
    return normalized_vol, recent_vol

@njit
def _calculate_kelly_criterion(
    data: NDArray[np.float64],
    slow_period: int = 60,
    fast_period: int = 10,
    kelly_fraction: float = 0.5
) -> Tuple[NDArray[np.float64], float]:
    """
    Compute Kelly position sizing based on fast/slow volatility ratio.

    Returns:
        - kelly_criterion: full array of sizing values
        - recent_criterion: latest value as float
    """
    n = len(data)
    log_returns = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        log_returns[i] = np.log(data[i] / data[i - 1])
    
    vol_f = np.zeros(n, dtype=np.float64)
    vol_s = np.zeros(n, dtype=np.float64)
    
    for i in range(slow_period - 1, n):
        r_slice = log_returns[i - slow_period + 1:i + 1]
        std = np.std(r_slice)
        vol_f[i] = std * np.sqrt(slow_period)
    
    for i in range(fast_period - 1, n):
        r_slice = log_returns[i - fast_period + 1:i + 1]
        std = np.std(r_slice)
        vol_s[i] = std * np.sqrt(fast_period)
    
    kelly_criterion = vol_s / vol_f * kelly_fraction

    return kelly_criterion

@njit
def estimate_q_r(price: np.ndarray, window: int = 60) -> Tuple[float, float]:
    n = len(price)
    if n < window + 2:
        # Not enough data, fallback to simple variance of returns for both
        log_returns = np.diff(np.log(price))
        var_lr = np.var(log_returns) if len(log_returns) > 0 else 1e-6
        return var_lr, var_lr

    log_returns = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        log_returns[i - 1] = np.log(price[i] / price[i - 1])

    # Observation noise R: variance of last 'window' log returns
    mean_lr = 0.0
    for i in range(n - window - 1, n - 1):
        mean_lr += log_returns[i]
    mean_lr /= window

    R = 0.0
    for i in range(n - window - 1, n - 1):
        R += (log_returns[i] - mean_lr) ** 2
    R /= window

    # Process noise Q: variance of diff of log returns (acceleration)
    process_noise_len = n - 2
    process_noise = np.empty(process_noise_len, dtype=np.float64)
    for i in range(1, n - 1):
        process_noise[i - 1] = log_returns[i] - log_returns[i - 1]

    mean_pn = 0.0
    for i in range(process_noise_len - window, process_noise_len):
        mean_pn += process_noise[i]
    mean_pn /= window

    Q = 0.0
    for i in range(process_noise_len - window, process_noise_len):
        Q += (process_noise[i] - mean_pn) ** 2
    Q /= window

    # Prevent zero variance (which breaks Kalman)
    if Q == 0.0:
        Q = 1e-8
    if R == 0.0:
        R = 1e-8

    return Q, R

@njit
def _kalman_filter_mean(
    data: NDArray[np.float64],
    process_variance: float,
    measurement_variance: float,
    adjust_factor: float = 1.0
) -> NDArray[np.float64]:
    """
    Kalman Filter to estimate dynamic mean of a time series.
    
    Args:
        data: Input price series.
        process_variance: Q - variance of the process (smoother = smaller).
        measurement_variance: R - variance of the observations (noisier = larger).
    
    Returns:
        Filtered series as dynamic mean estimates.
    """
    n = len(data)
    estimated = np.empty(n, dtype=np.float64)
    estimate = data[0]  # initial estimate
    error_estimate = 1.0  # initial error estimate

    for t in range(n):
        # Prediction update
        error_estimate += process_variance * adjust_factor

        # Measurement update (Kalman gain)
        kalman_gain = error_estimate / (error_estimate + measurement_variance)
        estimate = estimate + kalman_gain * (data[t] - estimate)
        error_estimate = (1 - kalman_gain) * error_estimate

        estimated[t] = estimate

    return estimated

def kalman_filter_mean(
    data: np.ndarray,
    process_variance: float = -1.0,
    measurement_variance: float = -1.0,
    adjust_factor: float = 1.0
) -> np.ndarray:
    """
    Wrapper for _kalman_filter_mean that estimates Q and R if not provided.
    """
    if process_variance < 0 or measurement_variance < 0:
        process_variance, measurement_variance = estimate_q_r(data)
    return _kalman_filter_mean(data, process_variance, measurement_variance, adjust_factor)

@njit
def _z_score(
    series: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    epsilon: float = 1e-8
) -> NDArray[np.float64]:
    """
    Compute Z-score: (series - mean) / std with epsilon to avoid division by zero.
    
    Args:
        series: Input data (e.g., price).
        mean: Smoothed mean (e.g., Kalman, EMA).
        std: Volatility estimate (e.g., rolling/EWM std).
        epsilon: Small constant to prevent divide-by-zero.
    
    Returns:
        Z-score array.
    """
    n = len(series)
    z = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if std[i] > epsilon:
            z[i] = (series[i] - mean[i]) / std[i]

    return z

@njit
def _cross_value(series: NDArray[np.float64], value: float) -> NDArray[np.float64]:
    """
    Detect cross-over and cross-under events of a series vs. a constant value.

    Args:
        series: Input time series (e.g., price or indicator).
        value: Constant threshold to check for crossing.

    Returns:
        NDArray[np.float64]: Array with non-zero values only at cross points:
            +diff → upward cross (series crosses above value)
            -diff → downward cross (series crosses below value)
            0     → no cross
    """
    n = len(series)
    output = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        prev = series[i - 1]
        curr = series[i]

        if prev < value and curr > value:
            output[i] = curr - value  # upward cross
        elif prev > value and curr < value:
            output[i] = curr - value  # downward cross (negative)
        else:
            output[i] = 0.0

    return output

@njit
def _cross(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    """
    Detect cross-over events between two series.

    Args:
        series_a: First input series (e.g., price).
        series_b: Second input series (e.g., moving average).

    Returns:
        np.ndarray: Array where:
            +1 = cross over (a crosses above b),
            -1 = cross under (a crosses below b),
             0 = no cross.
    """
    n = len(series_a)
    crosses = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        prev_a = series_a[i - 1]
        prev_b = series_b[i - 1]
        curr_a = series_a[i]
        curr_b = series_b[i]

        if prev_a < prev_b and curr_a > curr_b:
            crosses[i] = 1
        elif prev_a > prev_b and curr_a < curr_b:
            crosses[i] = -1
        else:
            crosses[i] = 0

    return crosses


class Indicator:
    macd = staticmethod(_macd)
    average_true_range = staticmethod(_average_true_range)
    rolling_std = staticmethod(rolling_std)
    moving_average = staticmethod(moving_average)
    bollinger_bands = staticmethod(_calculate_bollinger_bands)
    relative_strength_index = staticmethod(_rsi)
    volatility_normalize = staticmethod(_calculate_volatility_normalize)
    kelly_criterion = staticmethod(_calculate_kelly_criterion)
    kalman_filter_mean = staticmethod(kalman_filter_mean)    # wrapper for external use
    z_score = staticmethod(_z_score)
    cross = staticmethod(_cross)
    cross_value = staticmethod(_cross_value)
    trend_detect = staticmethod(_trend_detect)

