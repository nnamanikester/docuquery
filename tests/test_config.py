"""Settings: the config file, the setup questions, and validation.

The subprocess tests run in a fresh Python process. They check behaviour that
depends on import order, which a test inside this process cannot see.
"""

import os
import stat
import subprocess
import sys
import textwrap

import pytest
from click.testing import CliRunner

import docuquery.utils
from docuquery.main import cli
from docuquery.utils import (
    CONFIG_VARIABLES,
    check_config_file,
    ensure_env_variables,
    set_env_variable,
)

ALL_KEYS = [config.key for config in CONFIG_VARIABLES]


@pytest.fixture
def no_settings(monkeypatch):
    """Remove every setting from the environment, as for a new user."""
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


class FakeDatabase:
    def initialize_database(self):
        pass


@pytest.fixture
def fake_database(monkeypatch):
    monkeypatch.setattr("docuquery.database.Database", FakeDatabase)


def run_python(script, tmp_path, **extra_env):
    """Run a script in a fresh Python process, with HOME in a temporary folder."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        "APPDATA": str(tmp_path),
        **extra_env,
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# --- Config file ---------------------------------------------------------------


def test_config_file_values_are_loaded(isolated_config_file, monkeypatch):
    monkeypatch.delenv("DOCUQUERY_OPENAI_MODEL")
    isolated_config_file.parent.mkdir(parents=True)
    isolated_config_file.write_text("DOCUQUERY_OPENAI_MODEL=model-from-file\n")

    assert check_config_file() is True
    assert os.environ["DOCUQUERY_OPENAI_MODEL"] == "model-from-file"


def test_environment_variable_has_priority_over_the_file(
    isolated_config_file, monkeypatch
):
    monkeypatch.setenv("DOCUQUERY_OPENAI_MODEL", "model-from-env")
    isolated_config_file.parent.mkdir(parents=True)
    isolated_config_file.write_text("DOCUQUERY_OPENAI_MODEL=model-from-file\n")

    check_config_file()
    assert os.environ["DOCUQUERY_OPENAI_MODEL"] == "model-from-env"


def test_missing_config_file_is_not_an_error():
    assert check_config_file() is False


def test_saved_setting_is_written_to_the_file(isolated_config_file, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    set_env_variable("OPENAI_API_KEY", "sk-saved")

    assert os.environ["OPENAI_API_KEY"] == "sk-saved"
    assert isolated_config_file.read_text() == "OPENAI_API_KEY=sk-saved\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file permissions")
def test_config_file_is_private(isolated_config_file, monkeypatch):
    # The file holds the API key, so only the owner may read it
    monkeypatch.delenv("OPENAI_API_KEY")
    set_env_variable("OPENAI_API_KEY", "sk-saved")
    assert stat.S_IMODE(isolated_config_file.stat().st_mode) == 0o600


# --- Setup questions -----------------------------------------------------------


@pytest.mark.usefixtures("no_settings", "fake_database")
def test_new_user_is_asked_and_the_answers_are_saved(isolated_config_file):
    answers = "postgresql://u:p@db.example.com/docs\nsk-secret\n\n\n\n\n"
    result = CliRunner().invoke(cli, ["init"], input=answers)

    assert result.exit_code == 0, result.output
    saved = dict(
        line.split("=", 1) for line in isolated_config_file.read_text().splitlines()
    )
    assert saved == {
        "DOCUQUERY_POSTGRES_URL": "postgresql://u:p@db.example.com/docs",
        "OPENAI_API_KEY": "sk-secret",
        "DOCUQUERY_OPENAI_MODEL": "gpt-5.4-mini",
        "SENTENCE_TRANSFORMER_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "DOCUQUERY_CHUNK_SIZE": "512",
        "DOCUQUERY_OVERLAP": "50",
    }


@pytest.mark.usefixtures("no_settings", "fake_database")
def test_api_key_is_not_shown_while_typed():
    answers = "postgresql://u:p@localhost/docs\nsk-secret\n\n\n\n\n"
    result = CliRunner().invoke(cli, ["init"], input=answers)
    assert "sk-secret" not in result.output
    assert "postgresql://u:p@localhost/docs" in result.output  # other answers echo


@pytest.mark.usefixtures("no_settings", "fake_database")
def test_saved_answers_are_not_asked_again(isolated_config_file):
    isolated_config_file.parent.mkdir(parents=True)
    isolated_config_file.write_text("".join(f"{key}=saved-value\n" for key in ALL_KEYS))
    result = CliRunner().invoke(cli, ["init"], input="")
    assert result.exit_code == 0, result.output
    assert "Missing" not in result.output


def test_no_questions_when_all_settings_are_set(monkeypatch):
    asked = []
    monkeypatch.setattr(
        docuquery.utils.click, "prompt", lambda *a, **_kwargs: asked.append(a)
    )
    ensure_env_variables()
    assert asked == []


@pytest.mark.usefixtures("no_settings")
def test_help_does_not_ask_questions(monkeypatch):
    asked = []
    monkeypatch.setattr(
        docuquery.utils.click, "prompt", lambda *a, **_kwargs: asked.append(a)
    )
    monkeypatch.setattr(sys, "argv", ["docuquery", "ingest", "--help"])
    ensure_env_variables()
    assert asked == []


# --- Import order (fresh process) ----------------------------------------------


def test_config_file_values_reach_the_app(tmp_path):
    # Regression (N2): main.py imported constants.py before the config file was
    # loaded, so saved settings were ignored and the defaults were used.
    result = run_python(
        """
        import sys
        from pathlib import Path

        import click

        config = Path(click.get_app_dir("docuquery")) / "config.env"
        config.parent.mkdir(parents=True)
        config.write_text(
            "DOCUQUERY_POSTGRES_URL=postgresql://from-file/db\\n"
            "OPENAI_API_KEY=sk-file\\n"
            "DOCUQUERY_OPENAI_MODEL=model-from-file\\n"
            "SENTENCE_TRANSFORMER_MODEL=st-from-file\\n"
            "DOCUQUERY_CHUNK_SIZE=300\\n"
            "DOCUQUERY_OVERLAP=10\\n"
        )

        import docuquery.main

        sys.argv = ["docuquery", "sessions"]
        docuquery.main.cli.callback()

        import docuquery.constants as c

        print(c.DOCUQUERY_POSTGRES_URL, c.DOCUQUERY_OPENAI_MODEL, c.CHUNK_SIZE, c.OVERLAP)
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "postgresql://from-file/db",
        "model-from-file",
        "300",
        "10",
    ]


def test_importing_the_cli_loads_no_settings_and_no_torch(tmp_path):
    # Keeps --help and --version fast, and keeps N2 from coming back
    result = run_python(
        """
        import sys
        import docuquery.main
        print("docuquery.constants" in sys.modules, "torch" in sys.modules)
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


# --- Validation (fresh process) ------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        ("abc", "50", "DOCUQUERY_CHUNK_SIZE must be a whole number"),
        ("512", "ten", "DOCUQUERY_OVERLAP must be a whole number"),
        ("512", "512", "must be 0 or more, and less than DOCUQUERY_CHUNK_SIZE"),
        ("512", "-1", "must be 0 or more, and less than DOCUQUERY_CHUNK_SIZE"),
    ],
)
def test_invalid_number_settings_stop_with_a_clear_message(
    tmp_path, chunk_size, overlap, message
):
    result = run_python(
        "import docuquery.constants",
        tmp_path,
        DOCUQUERY_CHUNK_SIZE=chunk_size,
        DOCUQUERY_OVERLAP=overlap,
    )
    assert result.returncode == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_valid_number_settings_are_accepted(tmp_path):
    result = run_python(
        "import docuquery.constants as c; print(c.CHUNK_SIZE, c.OVERLAP)",
        tmp_path,
        DOCUQUERY_CHUNK_SIZE="256",
        DOCUQUERY_OVERLAP="0",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["256", "0"]
