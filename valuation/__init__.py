"""
valuation - 实时估值模块 (原 core.py + providers.py)
"""
from .core import (
    load_state, save_state, validate_state,
    calculate_valuation, calculate_valuation_batch,
    calculate_valuation_by_state, _is_market_closed,
)
from .providers import (
    get_fund_name, set_etf_link_target, get_etf_link_target,
    clear_etf_link_target, refresh_stale_holdings,
    get_fund_nav_history, get_holdings, get_quotes,
    get_fund_5day_change, get_etf_realtime_change,
)