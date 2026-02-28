"""
grid - 低频网格交易策略模块 (原 strategy.py)
"""
from .config import (
    get_signal_history, backfill_signal_outcomes, calc_signal_win_rate,
    get_vol_sensitivity_info, update_vol_sensitivity, clear_vol_sensitivity,
    auto_calibrate_vol_sensitivity,
    set_market_regime, get_market_regime_info,
    get_fitness_scores, get_fund_fitness,
)
from .engine import generate_signal, generate_all_signals