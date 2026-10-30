import json
from unittest.mock import MagicMock

import pytest

from docuquery.rag import RAGBase
from docuquery.tools import TOOLS

SESSION_ID = "3f6c2a1e-8d4b-4c1a-9e2f-7b5d0a9c1e33"


@pytest.fixture
def rag(fake_embedder):
    rag = RAGBase()
    rag.db = MagicMock()
    rag.db.search_similar_chunks.return_value = [
        {
            "file_name": "notes.txt",
            "content": "The secret code is 4417.",
            "similarity": 0.9,
        }
    ]
    rag.embedder = fake_embedder
    return rag


def answer(rag, client, question="What is the code?"):
    rag.openai_client = client
    return "".join(rag.query_rag_stream(question))


def assert_valid_history(messages):
    """Every assistant tool call must be followed by a tool reply with its ID.

    If not, OpenAI rejects every later request in the conversation.
    """
    for i, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            replies = [
                m for m in messages[i + 1 :] if m.get("tool_call_id") == call["id"]
            ]
            assert replies, f"tool call {call['id']} has no tool reply"


# --- Streaming -----------------------------------------------------------------


def test_streams_a_plain_answer(rag, llm):
    client = llm.client([llm.text("The code is 4417.")])
    assert answer(rag, client) == "The code is 4417."
    assert rag.messages[-1] == {"role": "assistant", "content": "The code is 4417."}


def test_sends_the_model_and_the_tools(rag, llm):
    client = llm.client([llm.text("ok")])
    answer(rag, client)
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["tools"] == TOOLS
    assert request["stream"] is True


def test_ignores_chunks_without_choices(rag, llm):
    # Regression: usage-only and content-filter chunks have an empty choices list
    client = llm.client([[llm.empty_chunk(), *llm.text("still works")]])
    assert answer(rag, client) == "still works"


# --- Tool calls ----------------------------------------------------------------


def test_runs_a_search_and_sends_the_result_back(rag, llm):
    client = llm.client(
        [llm.tool(("call_1", "search", '{"query": "secret code"}')), llm.text("4417")]
    )
    assert answer(rag, client) == "4417"

    tool_message = rag.messages[-2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "The secret code is 4417." in tool_message["content"]
    # The second request carries the tool result to the model
    assert client.requests[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert_valid_history(rag.messages)


def test_runs_parallel_tool_calls_in_order(rag, llm):
    client = llm.client(
        [
            llm.tool(
                ("call_a", "search", '{"query": "a"}'),
                ("call_b", "search", '{"query": "b"}'),
            ),
            llm.text("done"),
        ]
    )
    answer(rag, client)
    tool_ids = [m["tool_call_id"] for m in rag.messages if m["role"] == "tool"]
    assert tool_ids == ["call_a", "call_b"]
    assert_valid_history(rag.messages)


def test_malformed_tool_arguments_go_back_to_the_model(rag, llm):
    # Regression: invalid JSON used to crash and leave the history invalid
    client = llm.client(
        [llm.tool(("call_1", "search", '{"query": ')), llm.text("recovered")]
    )
    assert answer(rag, client) == "recovered"
    tool_message = next(m for m in rag.messages if m["role"] == "tool")
    assert tool_message["content"].startswith("Tool error:")
    assert_valid_history(rag.messages)


def test_a_failing_tool_goes_back_to_the_model(rag, llm):
    rag.db.search_similar_chunks.side_effect = RuntimeError("database is down")
    client = llm.client(
        [llm.tool(("call_1", "search", '{"query": "x"}')), llm.text("sorry")]
    )
    assert answer(rag, client) == "sorry"
    tool_message = next(m for m in rag.messages if m["role"] == "tool")
    assert tool_message["content"] == "Tool error: database is down"
    assert_valid_history(rag.messages)


def test_unknown_tool_is_reported_to_the_model(rag, llm):
    client = llm.client([llm.tool(("call_1", "delete_all", "{}")), llm.text("ok")])
    answer(rag, client)
    tool_message = next(m for m in rag.messages if m["role"] == "tool")
    assert tool_message["content"] == "Unknown tool: delete_all"


def test_stops_after_the_maximum_number_of_tool_turns(rag, llm):
    turn = llm.tool(("call_1", "search", '{"query": "x"}'))
    client = llm.client([turn, turn, turn])
    assert answer(rag, client).endswith("[Stopped: reached max tool-call turns]")
    assert len(client.requests) == 3
    assert_valid_history(rag.messages)


# Minimal valid arguments for each tool the model is offered
TOOL_ARGUMENTS = {
    "search": {"query": "x"},
    "__count_documents": {"query": "x"},
    "__list_documents": {},
    "list_sessions": {},
    "get_session_messages": {"session_id": SESSION_ID},
}


def test_every_offered_tool_has_arguments_in_this_test():
    assert {t["function"]["name"] for t in TOOLS} == set(TOOL_ARGUMENTS)


@pytest.mark.parametrize("name", list(TOOL_ARGUMENTS))
def test_every_offered_tool_runs(rag, llm, name):
    rag.db.count_documents.return_value = 0
    rag.db.list_documents.return_value = []
    rag.db.list_conversations.return_value = []
    rag.db.get_conversation_by_id.return_value = None
    client = llm.client(
        [llm.tool(("call_1", name, json.dumps(TOOL_ARGUMENTS[name]))), llm.text("ok")]
    )
    answer(rag, client)
    tool_message = next(m for m in rag.messages if m["role"] == "tool")
    assert not tool_message["content"].startswith(("Unknown tool", "Tool error"))


# --- Search --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "used"), [(-1, 1), (0, 1), (2, 2), (20, 20), (500, 20)]
)
def test_search_limit_is_kept_between_1_and_20(rag, requested, used):
    rag.search("x", limit=requested)
    assert rag.db.search_similar_chunks.call_args.kwargs["limit"] == used


def test_build_context_labels_each_field(rag):
    context = rag.build_context(
        [{"file_name": "a.txt", "content": "Hello", "relevance": None}]
    )
    assert "File Name: a.txt" in context
    assert "Content: Hello" in context
    assert "Relevance" not in context  # None values are left out


def test_build_context_with_no_matches(rag):
    assert rag.build_context([]) == "No relevant context found."


# --- Sessions ------------------------------------------------------------------


def test_first_save_does_not_store_the_system_prompt(rag, llm):
    rag.db.create_conversation.return_value = SESSION_ID
    answer(rag, llm.client([llm.text("4417")]), question="What is the code?")

    assert rag.save_conversation(title="What is the code?") == SESSION_ID

    saved = rag.db.add_messages.call_args.kwargs["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"]


def test_later_saves_store_only_new_messages(rag, llm):
    rag.db.create_conversation.return_value = SESSION_ID
    answer(rag, llm.client([llm.text("one")]), question="first")
    rag.save_conversation(title="first")
    answer(rag, llm.client([llm.text("two")]), question="second")
    rag.save_conversation(title="first")

    rag.db.create_conversation.assert_called_once()
    saved = rag.db.add_messages.call_args.kwargs["messages"]
    assert [m["content"] for m in saved] == ["second", "two"]


def test_loading_a_session_gives_one_system_prompt(rag):
    # Regression: the stored system prompt plus a new one gave two system messages
    rag.db.get_conversation_by_id.return_value = {
        "title": "t",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    rag.load_conversation(SESSION_ID)
    assert [m["role"] for m in rag.messages] == ["system", "user", "assistant"]
    assert SESSION_ID in rag.messages[0]["content"]


def test_saving_after_loading_adds_only_new_messages(rag, llm):
    rag.db.get_conversation_by_id.return_value = {
        "title": "t",
        "messages": [{"role": "user", "content": "hi"}],
    }
    rag.load_conversation(SESSION_ID)
    answer(rag, llm.client([llm.text("new answer")]), question="new question")

    assert rag.save_conversation(title="ignored") == SESSION_ID
    rag.db.create_conversation.assert_not_called()
    saved = rag.db.add_messages.call_args.kwargs["messages"]
    assert [m["content"] for m in saved] == ["new question", "new answer"]


def test_loading_an_unknown_session_raises(rag):
    rag.db.get_conversation_by_id.return_value = None
    with pytest.raises(ValueError, match="No conversation found"):
        rag.load_conversation(SESSION_ID)


def test_session_messages_show_only_user_and_assistant(rag):
    rag.db.get_conversation_by_id.return_value = {
        "title": "t",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
            {"role": "tool", "content": "secret tool output", "tool_call_id": "c"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    assert rag.get_session_messages(SESSION_ID) == (
        "user: hi\nassistant: [tool call]\nassistant: hello"
    )


def test_session_messages_for_an_unknown_session(rag):
    rag.db.get_conversation_by_id.return_value = None
    assert rag.get_session_messages(SESSION_ID) == "Session does not exist."


def test_embedding_model_loads_only_when_needed(monkeypatch):
    # The sessions command must not load the model
    loaded = []
    monkeypatch.setattr(
        "docuquery.rag.SentenceTransformer", lambda *a: loaded.append(a)
    )
    RAGBase()
    assert loaded == []
