# financial/__init__.py

# Import from data subdirectory
from .data.minimal_pipeline import MinimalMarketPipeline

# Also import other modules you might need
try:
    from .financial_tools import FinancialTools
    from .auth.upstox_auth import UpstoxAuth, check_upstox_auth, upstox_auth_flow
except ImportError:
    # Optional imports might fail
    pass

__all__ = ['MinimalMarketPipeline', 'FinancialTools', 'UpstoxAuth']