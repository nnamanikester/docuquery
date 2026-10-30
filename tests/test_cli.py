from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from docuquery.main import cli

SESSION_ID = "3f6c2a1e-8d4b-4c1a-9e2f-7b5d0a9c1e33"


@pytest.fixture
def runner():
    return CliRunner()


class FakeRAG:
    """Stands in for RAGBase. Each answer is a string, or an exception to raise."""

    def __init__(self):
        self.answers = []
        self.questions = []
        self.saved_titles = []
        self.sessions = []
        self.conversations = {}

    def query_rag_stream(self, question):
        self.questions.append(question)
        result = self.answers.pop(0)
        if isinstance(result, Exception):
            raise result
        yield result

    def save_conversation(self, title):
        self.saved_titles.append(title)
        return SESSION_ID

    def load_conversation(self, session_id):
        if session_id not in self.conversations:
            raise ValueError(f"No conversation found for session_id: {session_id}")
        return self.conversations[session_id]

    def list_sessions(self):
        if isinstance(self.sessions, Exception):
            raise self.sessions
        return self.sessions


@pytest.fixture
def rag(monkeypatch):
    fake = FakeRAG()
    monkeypatch.setattr("docuquery.rag.RAGBase", lambda: fake)
    return fake


# --- General -------------------------------------------------------------------


def test_version(runner):
    result = runner.invoke(cli, ["--version"], prog_name="docuquery")
    assert result.exit_code == 0
    assert "docuquery, version" in result.output


def test_help_lists_the_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("init", "ingest", "ask", "sessions", "resume"):
        assert command in result.output
    assert "previous_sessions" not in result.output


# --- init ----------------------------------------------------------------------


def test_init_succeeds(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "docuquery.database.Database",
        lambda: SimpleNamespace(initialize_database=lambda: calls.append("init")),
    )
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "Database initialized successfully" in result.output
    assert calls == ["init"]


def test_init_failure_exits_with_1(runner, monkeypatch):
    def fail():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(
        "docuquery.database.Database", lambda: SimpleNamespace(initialize_database=fail)
    )
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 1
    assert "Could not initialize database: connection refused" in result.output


# --- ingest --------------------------------------------------------------------


def _patch_ingestor(monkeypatch, process):
    monkeypatch.setattr(
        "docuquery.ingest.DocumentIngestor",
        lambda: SimpleNamespace(process_and_store_document=process),
    )


def test_ingest_succeeds(runner, monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    processed = []
    _patch_ingestor(monkeypatch, processed.append)

    result = runner.invoke(cli, ["ingest", str(path)])

    assert result.exit_code == 0
    assert processed == [str(path)]


def test_ingest_shows_a_click_error_once(runner, monkeypatch, tmp_path):
    # Regression: messages once read "Error: ❌ Error ingesting document: ..."
    path = tmp_path / "empty.txt"
    path.write_text("")

    def reject(_path):
        raise click.ClickException("'empty.txt' has no text.")

    _patch_ingestor(monkeypatch, reject)
    result = runner.invoke(cli, ["ingest", str(path)])

    assert result.exit_code == 1
    assert result.output.strip() == "Error: 'empty.txt' has no text."


def test_ingest_wraps_other_errors(runner, monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    def crash(_path):
        raise RuntimeError("disk full")

    _patch_ingestor(monkeypatch, crash)
    result = runner.invoke(cli, ["ingest", str(path)])

    assert result.exit_code == 1
    assert "Could not ingest the document: disk full" in result.output


def test_ingest_rejects_a_missing_file(runner, tmp_path):
    result = runner.invoke(cli, ["ingest", str(tmp_path / "missing.pdf")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


# --- ask -----------------------------------------------------------------------


def test_ask_answers_and_saves_the_session(runner, rag):
    rag.answers = ["The code is 4417."]
    result = runner.invoke(cli, ["ask", "What is the code?"], input="exit\n")

    assert result.exit_code == 0
    assert "The code is 4417." in result.output
    assert rag.saved_titles == ["What is the code?"]
    assert "Goodbye!" in result.output


def test_ask_continues_the_conversation(runner, rag):
    rag.answers = ["first answer", "second answer"]
    result = runner.invoke(cli, ["ask", "first"], input="second\nquit\n")

    assert rag.questions == ["first", "second"]
    assert "second answer" in result.output
    # The title stays the first question
    assert rag.saved_titles == ["first", "first"]


def test_ask_without_a_question_prompts_for_one(runner, rag):
    rag.answers = ["hi there"]
    result = runner.invoke(cli, ["ask"], input="hello\nexit\n")
    assert rag.questions == ["hello"]
    assert "hi there" in result.output


def test_ask_title_is_cut_to_50_characters(runner, rag):
    rag.answers = ["ok"]
    runner.invoke(cli, ["ask", "q" * 80], input="exit\n")
    assert rag.saved_titles == ["q" * 50]


def test_ask_stops_at_the_end_of_input(runner, rag):
    rag.answers = ["ok"]
    result = runner.invoke(cli, ["ask", "question"], input="")
    assert result.exit_code == 0
    assert "Goodbye!" in result.output


def test_a_failed_answer_does_not_end_the_chat(runner, rag):
    rag.answers = [RuntimeError("rate limit"), "second try works"]
    result = runner.invoke(cli, ["ask", "first"], input="again\nexit\n")

    assert result.exit_code == 0
    assert "Could not generate an answer: rate limit" in result.output
    assert "second try works" in result.output
    assert rag.questions == ["first", "again"]


def test_ask_resumes_a_session(runner, rag):
    rag.conversations[SESSION_ID] = {"title": "Old chat", "messages": []}
    result = runner.invoke(cli, ["ask", "--session-id", SESSION_ID], input="exit\n")
    assert result.exit_code == 0
    assert f'Resumed session: {SESSION_ID} - Title: "Old chat"' in result.output


@pytest.mark.usefixtures("rag")
def test_ask_rejects_an_invalid_session_id(runner):
    result = runner.invoke(cli, ["ask", "--session-id", "abc"])
    assert result.exit_code == 1
    assert "Invalid session ID" in result.output


@pytest.mark.usefixtures("rag")
def test_ask_rejects_an_unknown_session(runner):
    result = runner.invoke(cli, ["ask", "--session-id", SESSION_ID])
    assert result.exit_code == 1
    assert "Session does not exist" in result.output


# --- resume --------------------------------------------------------------------


def test_resume_loads_the_session(runner, rag):
    # Regression: a valid session ID once crashed with a TypeError
    rag.conversations[SESSION_ID] = {"title": "Old chat", "messages": []}
    result = runner.invoke(cli, ["resume", SESSION_ID], input="exit\n")
    assert result.exit_code == 0
    assert "Resumed session" in result.output


@pytest.mark.usefixtures("rag")
def test_resume_rejects_an_invalid_session_id(runner):
    result = runner.invoke(cli, ["resume", "abc"])
    assert result.exit_code == 1
    assert "Invalid session ID" in result.output


@pytest.mark.usefixtures("rag")
def test_resume_rejects_an_unknown_session(runner):
    result = runner.invoke(cli, ["resume", SESSION_ID])
    assert result.exit_code == 1
    assert "Session does not exist" in result.output


# --- sessions ------------------------------------------------------------------


def test_sessions_lists_each_session(runner, rag):
    rag.sessions = [
        {"id": SESSION_ID, "title": "First chat"},
        {"id": "9a1b7c3d-2e4f-4a6b-8c0d-1e2f3a4b5c6d", "title": "Second chat"},
    ]
    result = runner.invoke(cli, ["sessions"])
    assert result.exit_code == 0
    assert f'Session ID: {SESSION_ID} - Title: "First chat"' in result.output
    assert "Second chat" in result.output


@pytest.mark.usefixtures("rag")
def test_sessions_with_no_sessions(runner):
    result = runner.invoke(cli, ["sessions"])
    assert result.exit_code == 0
    assert "No sessions found." in result.output


def test_sessions_failure_exits_with_1(runner, rag):
    rag.sessions = RuntimeError("connection refused")
    result = runner.invoke(cli, ["sessions"])
    assert result.exit_code == 1
    assert "Could not list sessions: connection refused" in result.output
