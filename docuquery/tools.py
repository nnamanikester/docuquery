search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the user's ingested documents for passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to match against document content.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of search results to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

count_documents_tool = {
    "type": "function",
    "function": {
        "name": "__count_documents",
        "description": "Searches for documents, and counts the total number of documents that match the query.\n\nReturns:\n    int: The total number of documents that match thequery.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to match against document names, content, and embeddings.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

list_documents_tool = {
    "type": "function",
    "function": {
        "name": "__list_documents",
        "description": "Lists documents, optionally filtering by a query string and supporting pagination.\n\nReturns:\n    list[dict]: A list of dictionaries containing document information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search query to filter document names and content.",
                },
                "page": {
                    "type": "number",
                    "description": "The page number for pagination.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of documents to return per page.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

list_sessions_tool = {
    "type": "function",
    "function": {
        "name": "list_sessions",
        "description": "Get the list of conversation sessions.\n\nReturns:\n    list[dict]: Returns a list of conversation sessions, each represented as a dictionary with 'id' and 'title' keys.",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "number",
                    "description": "The page number to retrieve. Defaults to 1.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of sessions to return per page. Defaults to 20.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

get_session_messages_tool = {
    "type": "function",
    "function": {
        "name": "get_session_messages",
        "description": "Retrieve the formatted conversation messages for a given session ID.\n\nReturns:\n    str: The formatted conversation messages for the session, or an error message if the session does not exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The ID of the conversation session to load.",
                }
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [
    search_tool,
    count_documents_tool,
    list_documents_tool,
    list_sessions_tool,
    get_session_messages_tool,
]
