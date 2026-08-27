from src import config


def test_project_root_exists():
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()


def test_runtime_config_values_exist():
    assert config.DB_HOST
    assert config.DB_PORT
    assert config.DB_NAME

    assert config.OLLAMA_BASE_URL
    assert config.OLLAMA_MODEL


def test_db_port_is_integer():
    assert isinstance(
        config.DB_PORT,
        int,
    )


def test_api_port_is_integer():
    assert isinstance(
        config.API_PORT,
        int,
    )


def test_cors_origins_is_non_empty_list():
    assert isinstance(
        config.CORS_ORIGINS,
        list,
    )

    assert len(
        config.CORS_ORIGINS
    ) >= 1


def test_ollama_url_uses_http():
    assert (
        config.OLLAMA_BASE_URL.startswith(
            "http://"
        )
        or
        config.OLLAMA_BASE_URL.startswith(
            "https://"
        )
    )