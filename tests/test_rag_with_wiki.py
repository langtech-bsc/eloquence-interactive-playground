import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch

from gradio_app.backend import rag_with_wiki as rag


class CombinedEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, documents, replies, question="What did the speaker mention, and what is it?"):
        messages = [
            {"role": "system", "content": "Responde en español. Use only documents."},
            {"role": "user", "content": question},
        ]
        original = deepcopy(messages)
        tools = [{"type": "function", "function": {"name": name}}
                 for name in sorted(rag.WIKIPEDIA_TOOLS)]
        with (
            patch.object(rag, "load_wikipedia_tools", AsyncMock(return_value=tools)),
            patch.object(rag, "call_selected_model", AsyncMock(side_effect=replies)) as model,
            patch.object(rag, "execute_wikipedia_tool", AsyncMock(return_value="Article evidence")) as execute,
        ):
            result = await rag.run_rag_with_wikipedia(
                model_config={}, messages=messages, documents=documents,
                wikipedia_url="http://wikipedia.test/mcp", max_model_calls=3,
            )
        self.assertEqual(messages, original)
        self.assertIsNone(result["error"])
        return result, model, execute

    async def test_local_answer_has_chunks_and_tools_without_search(self):
        result, model, execute = await self.run_graph(
            ["The speaker mentioned astronomy."],
            [{"role": "assistant", "content": "Astronomy [doc 1]."}],
        )
        self.assertEqual(result["final_answer"], "Astronomy [doc 1].")
        execute.assert_not_called()
        messages = model.call_args.kwargs["messages"]
        self.assertTrue(messages[0]["content"].startswith(rag.EVIDENCE_INSTRUCTIONS))
        self.assertIn("Request preferences:", messages[0]["content"])
        self.assertIn('"doc": 1, "text": "The speaker mentioned astronomy."', messages[-1]["content"])
        self.assertEqual(messages[-1]["content"].count("What did the speaker mention"), 1)

    async def test_wikipedia_tools_work_with_nonempty_local_context(self):
        tool_calls = [{
            "id": "search-1", "type": "function",
            "function": {"name": "search_wikipedia", "arguments": '{"query": "astronomy"}'},
        }]
        result, model, execute = await self.run_graph(
            ["The speaker mentioned astronomy."],
            [
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {"role": "assistant", "content": "Combined answer."},
            ],
        )
        self.assertEqual(result["iteration"], 2)
        execute.assert_awaited_once_with(
            wikipedia_url="http://wikipedia.test/mcp",
            tool_name="search_wikipedia", arguments={"query": "astronomy"},
        )
        messages = model.call_args.kwargs["messages"]
        self.assertIn("The speaker mentioned astronomy.", messages[1]["content"])
        self.assertEqual(messages[-1], {
            "role": "tool", "tool_call_id": "search-1", "content": "Article evidence",
        })

    async def test_empty_retrieval_can_answer_a_greeting_without_search(self):
        result, model, execute = await self.run_graph(
            [], [{"role": "assistant", "content": "Hello!"}], question="Hello",
        )
        self.assertEqual(result["final_answer"], "Hello!")
        self.assertIn("evidence only):\n[]", model.call_args.kwargs["messages"][-1]["content"])
        execute.assert_not_called()

    async def test_last_model_call_is_forced_to_answer_without_tools(self):
        tool_call = {
            "id": "search-1", "type": "function",
            "function": {"name": "search_wikipedia", "arguments": '{"query": "astronomy"}'},
        }
        result, model, execute = await self.run_graph(
            [],
            [
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "assistant", "content": None, "tool_calls": [{
                    **tool_call, "id": "search-2",
                }]},
                {"role": "assistant", "content": "Final answer."},
            ],
        )
        self.assertEqual(result["final_answer"], "Final answer.")
        self.assertIsNone(model.call_args.kwargs["tools"])
        self.assertIn("final model call", model.call_args.kwargs["messages"][-1]["content"])
        execute.assert_awaited_once()

    async def test_wikipedia_failure_is_returned_to_model(self):
        messages = [{"role": "user", "content": "Explain this using Wikipedia."}]
        tool_call = {
            "id": "summary-1", "type": "function",
            "function": {"name": "get_summary", "arguments": '{"title": "Example"}'},
        }
        replies = [
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "assistant", "content": "Wikipedia is temporarily unavailable."},
        ]
        tools = [{"type": "function", "function": {"name": name}}
                 for name in sorted(rag.WIKIPEDIA_TOOLS)]
        with (
            patch.object(rag, "load_wikipedia_tools", AsyncMock(return_value=tools)),
            patch.object(rag, "call_selected_model", AsyncMock(side_effect=replies)) as model,
            patch.object(
                rag, "execute_wikipedia_tool",
                AsyncMock(side_effect=RuntimeError("MCP timeout")),
            ) as execute,
        ):
            result = await rag.run_rag_with_wikipedia(
                model_config={}, messages=messages, documents=[],
                wikipedia_url="http://wikipedia.test/mcp", max_model_calls=3,
            )

        self.assertIsNone(result["error"])
        self.assertEqual(result["final_answer"], "Wikipedia is temporarily unavailable.")
        execute.assert_awaited_once()
        tool_message = model.call_args.kwargs["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertFalse(json.loads(tool_message["content"])["ok"])

    async def test_tool_discovery_failure_still_allows_an_answer(self):
        messages = [{"role": "user", "content": "Use Wikipedia."}]
        with (
            patch.object(
                rag, "load_wikipedia_tools",
                AsyncMock(side_effect=RuntimeError("MCP unavailable")),
            ),
            patch.object(
                rag, "call_selected_model",
                AsyncMock(return_value={
                    "role": "assistant",
                    "content": "Wikipedia is temporarily unavailable.",
                }),
            ) as model,
        ):
            result = await rag.run_rag_with_wikipedia(
                model_config={}, messages=messages, documents=[],
                wikipedia_url="http://wikipedia.test/mcp", max_model_calls=3,
            )

        self.assertIsNone(result["error"])
        self.assertEqual(result["final_answer"], "Wikipedia is temporarily unavailable.")
        self.assertEqual(model.call_args.kwargs["tools"], [])
        self.assertIn(
            "Wikipedia tool status",
            model.call_args.kwargs["messages"][0]["content"],
        )

    async def test_unapproved_tool_is_rejected_before_execution(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported Wikipedia tool"):
            await self.run_graph(["Local evidence"], [{
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "bad-1", "type": "function",
                    "function": {"name": "other_tool", "arguments": "{}"},
                }],
            }])

    def test_all_retrieved_chunks_survive_and_distances_are_logged(self):
        chunks = [
            {"text": "First", "metadata": {"position": 1}, "distance": 1.2},
            {"text": "Second", "metadata": {"position": 2}, "distance": 0.1},
        ]
        retriever = Mock()
        retriever.search_with_scores.return_value = chunks
        with self.assertLogs("retrievers.diagnostics", level="INFO") as logs:
            result = rag.retrieve_chunks(
                retriever=retriever, index_name="example", query="question", top_k=2,
            )
        self.assertEqual(result, chunks)
        diagnostic = json.loads(logs.output[0].split("retrieval: ", 1)[1])
        self.assertIsNone(diagnostic["threshold"])
        self.assertEqual(diagnostic["kept"], 2)
        self.assertEqual([c["distance"] for c in diagnostic["chunks"]], [1.2, 0.1])
        retriever.search_with_scores.assert_called_once()


if __name__ == "__main__":
    unittest.main()
