from __future__ import annotations

import pytest

from integrations.quantdinger_bridge.api_adapter import (
    QuantDingerAPIClient,
    QuantDingerBusinessError,
    QuantDingerHTTPError,
    _normalize_ohlcv,
)
from integrations.quantdinger_bridge.config import ApiSettings


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def mount(self, *_args, **_kwargs):
        return None


def test_normalize_ohlcv_aliases() -> None:
    frame = _normalize_ohlcv(
        [{"t": 1_700_000_000_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}],
        symbol="AAPL",
        market="USStock",
        timeframe="1D",
    )

    assert list(frame[["symbol", "market", "timeframe"]].iloc[0]) == ["AAPL", "USStock", "1D"]
    assert frame["close"].iloc[0] == 1.5


def test_request_unwraps_quantdinger_envelope() -> None:
    client = QuantDingerAPIClient(ApiSettings(base_url="http://qd/api", token="abc"))
    fake_session = FakeSession(FakeResponse(200, {"code": 1, "data": [{"x": 1}]}))
    client.session = fake_session

    assert client._request("GET", "/market/symbols/search") == [{"x": 1}]
    assert fake_session.calls[0]["headers"]["Authorization"] == "Bearer abc"


def test_request_raises_business_error() -> None:
    client = QuantDingerAPIClient(ApiSettings(base_url="http://qd/api"))
    client.session = FakeSession(FakeResponse(200, {"code": 0, "msg": "bad request"}))

    with pytest.raises(QuantDingerBusinessError):
        client._request("GET", "/bad")


def test_request_raises_http_error() -> None:
    client = QuantDingerAPIClient(ApiSettings(base_url="http://qd/api"))
    client.session = FakeSession(FakeResponse(500, {"message": "boom"}))

    with pytest.raises(QuantDingerHTTPError):
        client._request("GET", "/bad")

