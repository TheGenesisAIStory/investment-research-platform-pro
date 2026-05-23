"""Expected-return prediction API."""

from .expected_returns import FundamentalPredictorRF, ExpectedReturnModel, describe_temporal_split, oos_r2, temporal_train_test_split

__all__ = ["ExpectedReturnModel", "FundamentalPredictorRF", "describe_temporal_split", "oos_r2", "temporal_train_test_split"]
