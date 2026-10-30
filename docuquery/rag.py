import json
import typing
from functools import cached_property

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from docuquery.constants import (
    DOCUQUERY_OPENAI_MODEL,
    OPENAI_API_KEY,
    SENTENCE_TRANSFORMER_MODEL,
    SYSTEM_INSTRUCTIONS,
    USER_PROMPT_TEMPLATE,
)
from docuquery.database import Database
from docuquery.tools import TOOLS


class RAGBase:
    MAX_SEARCH_RESULTS = 20

    def __init__(
        self,
        instructions: str = SYSTEM_INSTRUCTIONS,
        prompt_template: str = USER_PROMPT_TEMPLATE,
        model: str = DOCUQUERY_OPENAI_MODEL,
    ) -> None:
        self.db = Database()
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

        self.messages = [{"role": "system", "content": self.instructions}]
        self.saved_count = 1
        self.session_id = None

    @cached_property
    def embedder(self) -> SentenceTransformer:
        return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    def save_conversation(self, title: str) -> str:
        self.session_id = self.__start_new_session(title)
        self.db.add_messages(
            conversation_id=self.session_id, messages=self.messages[self.saved_count :]
        )
        self.saved_count = len(self.messages)
        return self.session_id

    def load_conversation(self, session_id: str) -> dict:
        """Load a conversation session by its session ID.

        Args:
            session_id (str): The ID of the conversation session to load.
        Raises:
            ValueError: If no conversation is found for the given session ID.
        Returns:
            dict: The conversation session data, including its messages.
        """
        conversation = self.db.get_conversation_by_id(conversation_id=session_id)
        if conversation is None:
            raise ValueError(f"No conversation found for session_id: {session_id}")
        message_content = self.instructions + "\n\n Session ID: " + session_id
        self.messages = [{"role": "system", "content": message_content}]
        self.messages.extend(conversation["messages"])
        self.session_id = session_id
        self.saved_count = len(self.messages)
        return conversation

    def list_sessions(self, page: int = 1, limit: int = 20) -> list[dict]:
        """Get the list of conversation sessions

        Args:
            page (int, optional): The page number to retrieve. Defaults to 1.
            limit (int, optional): The number of sessions per page. Defaults to 20.

        Returns:
            list[dict]: Returns a list of conversation sessions, each represented as a dictionary with 'id' and 'title' keys.
        """
        return self.db.list_conversations(page=page, page_size=limit)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        limit = (
            1
            if limit < 1
            else self.MAX_SEARCH_RESULTS
            if limit > self.MAX_SEARCH_RESULTS
            else limit
        )
        query_embedding = self.embedder.encode(query)
        return self.db.search_similar_chunks(
            query_embedding=query_embedding, limit=limit
        )

    def get_session_messages(self, session_id: str) -> str:
        """Retrieve the formatted conversation messages for a given session ID.

        Args:
            session_id (str): The ID of the conversation session to load.

        Returns:
            str: The formatted conversation messages for the session, or an error message if the session does not exist.
        """
        conversation = self.db.get_conversation_by_id(conversation_id=session_id)
        if conversation is None:
            return "Session does not exist."
        return "\n".join(
            f"{m['role']}: {m.get('content') or '[tool call]'}"
            for m in conversation["messages"]
            if m["role"] in ("user", "assistant")
        )

    def build_context(self, matches: list[dict]) -> str:

        # Loop through the matches and convert the dictionary into formated strings assuming the keys are unknown.
        context_str = ""

        if not matches:
            return "No relevant context found."

        for match in matches:
            for key in match:
                if match[key] is None:
                    continue
                field = key.split("_")
                field = " ".join([word.capitalize() for word in field])
                match[key] = f"{field}: {match[key]}"
                context_str += f"{match[key]}\n"

            context_str += "\n\n---\n\n"

        return context_str

    def build_prompt(self, matches: list[dict]) -> str:
        context = self.build_context(matches)
        return self.prompt_template.format(context=context)

    def __run_tool(self, name: str, args: dict) -> str:
        if name == "search":
            matches = self.search(args["query"], limit=args.get("limit", 3))
            return self.build_context(matches)
        if name == "__count_documents":
            return str(self.__count_documents(**args))
        if name == "__list_documents":
            matches = self.__list_documents(**args)
            return self.build_context(matches)
        if name == "list_sessions":
            sessions = self.list_sessions(**args)
            return self.build_context(sessions)
        if name == "get_session_messages":
            return self.get_session_messages(**args)
        return f"Unknown tool: {name}"

    def query_rag_stream(
        self, prompt: str, max_turns: int = 3
    ) -> typing.Generator[str, None, None]:
        self.messages.append({"role": "user", "content": prompt})

        for _ in range(max_turns):
            # Stream response tokens from OpenAI
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
                temperature=0.0,
                tools=TOOLS,
            )

            text_parts = []
            tool_calls = {}
            finish_reason = None

            for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield delta.content

                for tc in delta.tool_calls or []:
                    call = tool_calls.setdefault(
                        tc.index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc.id:
                        call["id"] = tc.id
                    if tc.function and tc.function.name:
                        call["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        call["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if finish_reason != "tool_calls":
                self.messages.append(
                    {"role": "assistant", "content": "".join(text_parts) or None}
                )
                return

            calls = [tool_calls[i] for i in sorted(tool_calls)]
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"],
                            },
                        }
                        for c in calls
                    ],
                }
            )

            for call in calls:
                function_name = call["name"]
                try:
                    args = json.loads(call["arguments"] or "{}")
                    result = self.__run_tool(function_name, args)
                except Exception as e:
                    result = (
                        f"Tool error: {e}"  # let the model see the failure and recover
                    )
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )

        yield "\n[Stopped: reached max tool-call turns]"

    def __count_documents(self, query: str) -> int:
        """Searches for documents, and counts the total number of documents that match the query.

        Args:
            query (str): The search query to match against document names, content, and embeddings.

        Returns:
            int: The total number of documents that match the query.
        """
        filters = {}
        if query.strip():
            filters["query_embedding"] = self.embedder.encode(query)
        return self.db.count_documents(**filters)

    def __list_documents(
        self, query: str = "", page: int = 1, limit: int = 5
    ) -> list[dict]:
        """Lists documents, optionally filtering by a query string and supporting pagination.

        Args:
            query (str): Optional search query to filter document names and content.
            limit (int): Maximum number of documents to return per page.
            page (int): The page number for pagination.

        Returns:
            list[dict]: A list of dictionaries containing document information.
        """
        filters = {}
        if query.strip():
            filters["query_embedding"] = self.embedder.encode(query)
        return self.db.list_documents(
            page=page,
            page_size=limit,
            **filters,
        )

    def __start_new_session(self, title: str) -> str:
        if self.session_id:
            return self.session_id

        self.session_id = self.db.create_conversation(title=title)
        return self.session_id
