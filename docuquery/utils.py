import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass
class ConfigVariable:
    key: str
    name: str
    desc: str
    default: str = ""
    secret: bool = False
    required: bool = False


CONFIG_FILE = Path(click.get_app_dir("docuquery")) / "config.env"

CONFIG_VARIABLES: list[ConfigVariable] = [
    ConfigVariable(
        key="DOCUQUERY_POSTGRES_URL",
        name="Postgres URL",
        desc="Set the Postgres URL for the Docuquery database",
        default="postgresql://postgres:postgres@localhost:5432/docuquery_db",
        required=True,
    ),
    ConfigVariable(
        key="OPENAI_API_KEY",
        name="OpenAI API Key",
        desc="Get your OpenAI API Key from https://platform.openai.com/account/api-keys",
        default="",
        secret=True,
        required=True,
    ),
    ConfigVariable(
        key="DOCUQUERY_OPENAI_MODEL",
        name="OpenAI Model",
        desc="See list of models you can use: https://platform.openai.com/docs/models",
        default="gpt-5.4-mini",
        required=False,
    ),
    ConfigVariable(
        key="SENTENCE_TRANSFORMER_MODEL",
        name="Sentence Transformer Model",
        desc="See list of models you can use: https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models",
        default="sentence-transformers/all-MiniLM-L6-v2",
        required=False,
    ),
    ConfigVariable(
        key="DOCUQUERY_CHUNK_SIZE",
        name="Chunk Size",
        desc="Set the chunk size for document processing",
        default="512",
        required=False,
    ),
    ConfigVariable(
        key="DOCUQUERY_OVERLAP",
        name="Overlap",
        desc="Set the overlap size for document processing",
        default="50",
        required=False,
    ),
]


def check_env_variable(name: str) -> bool:
    """Checks if an environment variable is set."""
    return bool(os.getenv(name, ""))


def set_env_variable(name: str, value: str) -> None:
    """Sets an environment variable."""
    os.environ[name] = value
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("a") as f:
        f.write(f"{name}={value}\n")
    CONFIG_FILE.chmod(0o600)


def check_config_file() -> bool:
    """Checks if the config file exists and loads environment variables from it."""
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        return True
    return False


def ensure_env_variables() -> None:
    if "--help" in sys.argv:
        return

    check_config_file()

    for config in CONFIG_VARIABLES:
        if check_env_variable(config.key):
            continue

        click.secho(
            f"Missing{' required' if config.required else ''} config: {config.name}",
            fg=f"{'red' if config.required else 'yellow'}",
        )
        if config.desc:
            click.secho(config.desc, err=True, fg="blue")

        while True:
            value = click.prompt(
                click.style(f"◉ {config.name}", fg="cyan"),
                hide_input=config.secret or False,
                prompt_suffix=" > ",
                type=str,
                default=config.default if not config.secret else None,
            )

            if value:
                set_env_variable(config.key, value)
                break
