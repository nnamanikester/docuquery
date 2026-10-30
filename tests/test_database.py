"""Database tests. They need a real PostgreSQL with pgvector.

They run only when DOCUQUERY_TEST_POSTGRES_URL is set. Each test deletes all
DocuQuery tables and the vector extension in that database, so never point it
at a database with data you want to keep. For example:

    docker run -d --rm --name docuquery-test -e POSTGRES_PASSWORD=test \
      -p 55432:5432 pgvector/pgvector:pg17
    DOCUQUERY_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:55432/postgres \
      uv run pytest
"""

import os
import uuid

import numpy as np
import psycopg
import pytest

from docuquery.database import Database

TEST_URL = os.environ.get("DOCUQUERY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not TEST_URL, reason="set DOCUQUERY_TEST_POSTGRES_URL to run the database tests"
)


def _reset(url):
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS conversation_messages, conversations, document_chunks"
        )
        conn.execute("DROP EXTENSION IF EXISTS vector CASCADE")
        conn.execute("DROP EXTENSION IF EXISTS pg_trgm CASCADE")


@pytest.fixture
def empty_database():
    """A database with no DocuQuery tables and no extensions, like a new one."""
    _reset(TEST_URL)
    return Database()


@pytest.fixture
def db(empty_database):
    empty_database.initialize_database()
    return empty_database


def vector(*hot):
    """A 384-dimension vector with 1.0 at the given positions."""
    v = np.zeros(384)
    v[list(hot)] = 1.0
    return v


def chunks_in(file_name):
    with psycopg.connect(TEST_URL) as conn:
        return conn.execute(
            "SELECT chunk_index, content FROM document_chunks"
            " WHERE file_name = %s ORDER BY chunk_index",
            (file_name,),
        ).fetchall()


# --- Setup ---------------------------------------------------------------------


def test_init_works_on_a_new_database(empty_database):
    # Regression (N1): init once failed with "vector type not found"
    empty_database.initialize_database()
    assert empty_database.list_documents() == []


def test_init_can_run_again(db):
    db.replace_document("a.txt", ["kept"], [vector(0)])
    db.initialize_database()
    assert chunks_in("a.txt") == [(0, "kept")]


# --- Documents -----------------------------------------------------------------


def test_replace_document_stores_the_chunks_in_order(db):
    db.replace_document("a.txt", ["one", "two"], [vector(0), vector(1)])
    assert chunks_in("a.txt") == [(0, "one"), (1, "two")]


def test_adding_a_file_again_replaces_it(db):
    # Regression: the same file added twice was stored twice
    db.replace_document("a.txt", ["old 1", "old 2", "old 3"], [vector(0)] * 3)
    db.replace_document("a.txt", ["new"], [vector(1)])
    assert chunks_in("a.txt") == [(0, "new")]


def test_replacing_one_file_keeps_the_others(db):
    db.replace_document("a.txt", ["a"], [vector(0)])
    db.replace_document("b.txt", ["b"], [vector(1)])
    db.replace_document("a.txt", ["a2"], [vector(2)])
    assert chunks_in("b.txt") == [(0, "b")]


def test_a_failed_replace_keeps_the_old_copy(db):
    # The delete and the insert are one transaction
    db.replace_document("a.txt", ["old"], [vector(0)])
    with pytest.raises(ValueError, match="zip"):
        db.replace_document("a.txt", ["one", "two"], [vector(1)])
    assert chunks_in("a.txt") == [(0, "old")]


def test_search_returns_the_nearest_chunks_first(db):
    db.replace_document("a.txt", ["about cats"], [vector(0)])
    db.replace_document("b.txt", ["about dogs"], [vector(1)])
    db.replace_document("c.txt", ["cats and dogs"], [vector(0, 1)])

    results = db.search_similar_chunks(vector(0), limit=2)

    assert [r["content"] for r in results] == ["about cats", "cats and dogs"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_list_and_count_documents(db):
    db.replace_document("a.txt", ["one", "two"], [vector(0), vector(1)])
    db.replace_document("b.txt", ["three"], [vector(2)])

    documents = db.list_documents()

    assert [(d["file_name"], d["chunk_count"]) for d in documents] == [
        ("a.txt", 2),
        ("b.txt", 1),
    ]
    assert db.count_documents() == 2


def test_list_documents_by_meaning(db):
    db.replace_document("cats.txt", ["cats"], [vector(0)])
    db.replace_document("dogs.txt", ["dogs"], [vector(1)])

    documents = db.list_documents(query_embedding=vector(0), sort_by="relevance")

    assert [d["file_name"] for d in documents] == ["cats.txt"]
    assert db.count_documents(query_embedding=vector(0)) == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"page": 0}, "page must be"),
        ({"page_size": 0}, "page_size must be"),
        ({"page_size": 101}, "page_size must be"),
        ({"sort_by": "content"}, "sort_by must be"),
        ({"sort_by": "relevance"}, "requires query_embedding"),
    ],
)
def test_list_documents_rejects_bad_arguments(db, arguments, message):
    with pytest.raises(ValueError, match=message):
        db.list_documents(**arguments)


# --- Conversations -------------------------------------------------------------


def test_conversation_round_trip_keeps_tool_calls(db):
    session_id = db.create_conversation(title="Test chat")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "x"}'},
        }
    ]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "hello"},
    ]
    db.add_messages(session_id, messages)

    conversation = db.get_conversation_by_id(session_id)

    assert conversation["title"] == "Test chat"
    # Empty fields are left out, so the messages go back to OpenAI unchanged
    assert conversation["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "hello"},
    ]


def test_messages_added_later_come_after_the_earlier_ones(db):
    session_id = db.create_conversation(title="t")
    db.add_messages(session_id, [{"role": "user", "content": "first"}])
    db.add_messages(session_id, [{"role": "user", "content": "second"}])

    contents = [m["content"] for m in db.get_conversation_by_id(session_id)["messages"]]
    assert contents == ["first", "second"]


def test_adding_messages_to_an_unknown_conversation_raises(db):
    with pytest.raises(ValueError, match="not found"):
        db.add_messages(str(uuid.uuid4()), [{"role": "user", "content": "hi"}])


def test_unknown_conversation_is_none(db):
    assert db.get_conversation_by_id(str(uuid.uuid4())) is None


def test_list_conversations_with_the_default_page_size(db):
    # Regression: page_size was compared with a list and every call raised TypeError
    db.create_conversation(title="only one")
    assert [c["title"] for c in db.list_conversations()] == ["only one"]


def test_list_conversations_shows_the_most_recent_first(db):
    older = db.create_conversation(title="older")
    db.create_conversation(title="newer")
    db.add_messages(older, [{"role": "user", "content": "makes it active again"}])

    titles = [c["title"] for c in db.list_conversations()]
    assert titles == ["older", "newer"]


def test_list_conversations_pages(db):
    for i in range(3):
        db.create_conversation(title=f"chat {i}")
    assert len(db.list_conversations(page=1, page_size=2)) == 2
    assert len(db.list_conversations(page=2, page_size=2)) == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"page": 0}, "page must be"),
        ({"page_size": 0}, "page_size must be"),
        ({"page_size": 101}, "page_size must be"),
    ],
)
def test_list_conversations_rejects_bad_arguments(db, arguments, message):
    with pytest.raises(ValueError, match=message):
        db.list_conversations(**arguments)
