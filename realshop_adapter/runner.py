"""Backward-compatible aliases for the renamed MerchantBench adapter."""

from merchantbench_adapter.runner import *  # noqa: F401,F403
from merchantbench_adapter.runner import (
    MerchantBenchHermesAgent as RealShopHermesAgent,
    MerchantBenchStaleStep as RealShopStaleStep,
    MerchantBenchToolClient as RealShopToolClient,
    MerchantBenchToolTurnComplete as RealShopToolTurnComplete,
)
