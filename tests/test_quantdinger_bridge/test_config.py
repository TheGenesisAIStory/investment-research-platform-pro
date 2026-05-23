from integrations.quantdinger_bridge.config import build_database_url, load_config


def test_load_config_defaults() -> None:
    config = load_config({})

    assert config.database.host == "postgres"
    assert config.database.dbname == "quantdinger"
    assert config.api.base_url == "http://backend:5000/api"
    assert config.service.api_prefix == "/api/v1/ml"


def test_load_config_env_overrides() -> None:
    config = load_config(
        {
            "QUANTDINGER_DB_HOST": "db",
            "QUANTDINGER_DB_PORT": "15432",
            "QUANTDINGER_DB_NAME": "qd",
            "QUANTDINGER_DB_USER": "alice",
            "QUANTDINGER_DB_PASSWORD": "secret",
            "QUANTDINGER_API_BASE_URL": "http://backend:5000/api/",
            "ML_SERVICE_PORT": "9000",
            "ML_SERVICE_API_PREFIX": "api/v2/ml",
        }
    )

    assert config.database.host == "db"
    assert config.database.port == 15432
    assert config.api.base_url == "http://backend:5000/api"
    assert config.service.port == 9000
    assert config.service.api_prefix == "/api/v2/ml"


def test_build_database_url_quotes_credentials() -> None:
    url = build_database_url("postgres", 5432, "quant dinger", "user@example.com", "p@ss word")

    assert "user%40example.com" in url
    assert "p%40ss+word" in url
    assert url.endswith("/quant+dinger")

