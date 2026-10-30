import os


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise SystemExit(
            f"{name} must be a whole number. The current value is '{value}'."
        ) from None


DOCUQUERY_POSTGRES_URL = os.getenv(
    "DOCUQUERY_POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/docuquery_db",
)
DOCUQUERY_OPENAI_MODEL = os.getenv("DOCUQUERY_OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


CHUNK_SIZE = _int_env("DOCUQUERY_CHUNK_SIZE", 512)
OVERLAP = _int_env("DOCUQUERY_OVERLAP", 50)

if not 0 <= OVERLAP < CHUNK_SIZE:
    raise SystemExit(
        "DOCUQUERY_OVERLAP must be 0 or more, and less than DOCUQUERY_CHUNK_SIZE."
    )

EMBEDDING_DIMENSION = 384  # must match SENTENCE_TRANSFORMER_MODEL's output size


SYSTEM_INSTRUCTIONS = """
You are an expert technical assistant. Answer the user's question accurately
using the provided tools. If the answer cannot be found,
say 'I cannot find the answer in the provided documents.'.

If you want to look up the documents, use the search tool. Feel free to do a multiple search
as needed to get the information you require from the documents.

There is a __list_documents tool that allows you to list documents with optional filtering and pagination.
Feel free to use it whenever you need to find documents. You can go over it, paginate, get more documents as needed.

There is a __count_documents tool that allows you to count the total number of documents matching a query.
Feel free to use it whenever you need to know how many documents match a specific search.

When using the search tool, provide a clear and concise query that will help retrieve the most relevant information.
If the user's question is about a specific document or set of documents, use the __list_documents tool first to identify them.
Then, use the search tool to find specific information within those documents.

User can ask about their session history and try to retrieve previous conversations using the list_sessions tool.

When a user requests for the session messages, you can use the get_session_messages tool to retrieve the messages for a specific session ID.

When call all theses tools. Ensure that you provide the necessary arguments and handle the responses appropriately. Do NOT provide
arguments that the functions cannot accept.

"""

USER_PROMPT_TEMPLATE = """

CONTEXT:
{context}
"""
