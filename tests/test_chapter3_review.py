"""Offline checks for the actual Chapter 3 notebook cells; no provider calls.

Install the pinned Lab 7/8 packages and nbformat, then run:
    python -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anthropic
import httpx2
import nbformat
from mistralai.client import Mistral

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "Chapter 3"


def notebook(lab):
    return json.loads(next(CHAPTER.glob(f"chapter_3_lab_{lab}_*.ipynb")).read_text())


def run_cell(lab, index, namespace):
    source = "".join(notebook(lab)["cells"][index]["source"])
    exec(compile(source, f"lab_{lab}_cell_{index}", "exec"), namespace)


def common():
    spec = importlib.util.spec_from_file_location("chapter3_common", CHAPTER / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Chapter3ReviewTests(unittest.TestCase):
    def test_notebook_format_and_python_syntax(self):
        for lab in (7, 8):
            data = notebook(lab)
            nbformat.validate(nbformat.from_dict(data))
            for index, cell in enumerate(data["cells"]):
                source = "".join(cell["source"])
                if cell["cell_type"] == "code" and not source.startswith("%pip"):
                    compile(source, f"lab_{lab}_cell_{index}", "exec")

    def test_lab7_request_and_response_through_real_sdk(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "msg_fixture", "type": "message", "role": "assistant",
                "model": "claude-sonnet-5", "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20},
                "content": [{"type": "text", "text": "Fixture memo: AAPL 29.3; JPM 11.8."}],
            })

        real_class = anthropic.Anthropic
        with httpx2.Client(transport=httpx2.MockTransport(respond)) as transport:
            client = real_class(api_key="offline-test-key", http_client=transport)
            ns = {"system_message": common().system_message, "input_message": common().input_message}
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test-key", "CHAPTER3_ANTHROPIC_MODEL": "claude-sonnet-5"}), patch("anthropic.Anthropic", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                for index in (14, 17, 19):
                    run_cell(7, index, ns)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request["model"], "claude-sonnet-5")
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertEqual(request["max_tokens"], 2000)
        self.assertNotIn("temperature", request)
        self.assertNotIn("tools", request)
        self.assertIn("Fixture memo", ns["memo"])

    def test_lab7_rejects_incomplete_or_empty_replies(self):
        for reason, content in (("max_tokens", "Partial memo"), ("refusal", "Cannot answer"), ("end_turn", "")):
            with self.subTest(reason=reason, content=content), self.assertRaises(RuntimeError):
                run_cell(7, 19, {"response": SimpleNamespace(stop_reason=reason, content=[SimpleNamespace(type="text", text=content)])})

    def test_lab7_reads_text_by_type_not_position(self):
        ns = {"response": SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Memo")])}
        with contextlib.redirect_stdout(io.StringIO()):
            run_cell(7, 19, ns)
        self.assertEqual(ns["memo"], "Memo")

    def test_lab8_manual_tool_loop_through_real_sdk(self):
        import httpx
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "stock0001", "type": "function", "function": {"name": "get_stock_data", "arguments": '{"ticker":"AAPL"}'}},
                    {"id": "ratio0001", "type": "function", "function": {"name": "compute_pe_ratio", "arguments": '{"price":195.3,"eps":6.67}'}},
                ]}
            else:
                message = {"role": "assistant", "content": "Fixture memo", "tool_calls": []}
            return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0, "model": "mistral-large-latest", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            client = Mistral(api_key="offline-test-key", client=transport)
            ns = vars(common()).copy()
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "offline-test-key"}), patch("mistralai.client.Mistral", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                for index in (14, 17, 20):
                    run_cell(8, index, ns)
        self.assertEqual(len(requests), 2)
        tool_messages = [m for m in requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertIn("195.3", tool_messages[0]["content"])
        self.assertIn("29.28", tool_messages[1]["content"])

    def test_scoring_script_reference(self):
        result = subprocess.run([sys.executable, str(CHAPTER / "templates/score_frameworks.py")], capture_output=True, text=True, check=True)
        self.assertIn("'Evaluation method at a glance' section, step 4", result.stdout)
        self.assertNotIn("Pilot template", result.stdout)


if __name__ == "__main__":
    unittest.main()
