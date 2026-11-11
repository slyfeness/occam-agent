"""Exercise the recorded tool middleware and asynchronous judge without models or external calls."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
from fastmcp import FastMCP

from src.stability_judge import FlashJudge
from src.stability_rollout import instrument_search_timing, join_tool_timings, read_lines, retrieval_timing, union_seconds


class FakeSearcher:
    def search(self, query):
        return [query]


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_timing_keeps_metadata_out_of_retrieval(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            mcp, searcher = FastMCP('test'), FakeSearcher()

            @mcp.tool(name='search')
            def search(query: str) -> list[str]:
                return searcher.search(query)

            module = instrument_search_timing(
                'import json,time\ndef start(mcp,searcher,folder):\n import hmac\n')
            namespace = {}
            exec(compile(module, 'middleware-fixture', 'exec'), namespace)
            namespace['start'](mcp, searcher, folder)
            result = await mcp._mcp_call_tool('search', {'query': 'fixture', '_occam_timing': 'opaque'})
            self.assertIn('fixture', result[0].text)
            rows = read_lines(folder / 'TOOL_TIMING.jsonl')
            self.assertEqual([row['trace'] for row in rows], ['opaque'])
            timing = rows[0]
            self.assertLessEqual(timing['handler_started'], timing['search_started'])
            self.assertLessEqual(timing['search_started'], timing['search_ended'])
            self.assertLessEqual(timing['search_ended'], timing['handler_ended'])

    async def test_api_verdict_attempts_four_times_and_keeps_payload(self):
        judge = FlashJudge.__new__(FlashJudge)
        judge.slots = asyncio.Semaphore(16)
        judge.protocol = {'parse_judge_response': lambda text: {'parse_error': False, 'correct': text == 'yes'}}
        requests = []

        async def send(request):
            requests.append(request.read())
            if len(requests) < 4:
                return httpx.Response(429)
            return httpx.Response(200, json={'model': 'fixture-model', 'choices': [
                {'finish_reason': 'stop', 'message': {'content': 'yes'}}]})

        async def no_delay(seconds):
            return None

        from unittest.mock import patch
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            judge.client = client
            with patch('src.stability_judge.asyncio.sleep', no_delay):
                row = await judge.verdict('fixture prompt')
        self.assertEqual(row['retries'], 3)
        self.assertIs(row['correct'], True)
        self.assertEqual(len(set(requests)), 1)

    def test_retrieval_wall_clock_does_not_double_count_parallel_calls(self):
        self.assertEqual(union_seconds([(1., 4.), (2., 5.), (8., 9.)]), 5.)

    def test_retrieval_rejected_before_search_keeps_measured_round_trip(self):
        client = {'trace': 'opaque', 'sent': 1., 'received': 4.}
        server = {'trace': 'opaque', 'handler_started': 1.5, 'handler_ended': 3.5}
        row = retrieval_timing(client, server)
        self.assertEqual(row['client_wall_seconds'], 3.)
        self.assertFalse(row['entered_search'])
        self.assertIsNone(row['search_seconds'])
        self.assertIsNone(row['server_queue_seconds'])

    def test_local_tool_rejection_does_not_consume_a_wire_request(self):
        events = [
            {'event_id': 1, 'dispatch_utc': '2026-01-01T00:00:00Z', 'return_utc': '2026-01-01T00:00:01Z'},
            {'event_id': 2, 'dispatch_utc': '2026-01-01T00:00:02Z', 'return_utc': '2026-01-01T00:00:05Z',
             'raw_mcp_response_path': 'private'},
        ]
        client = [{'trace': 'opaque', 'sent': 2., 'received': 5.}]
        server = {'opaque': {'trace': 'opaque', 'handler_started': 2.5, 'handler_ended': 4.5,
                             'search_started': 3., 'search_ended': 4.}}
        local, sent = join_tool_timings(events, client, server)
        self.assertFalse(local['dispatched_to_server'])
        self.assertEqual(local['event_wall_seconds'], 1.)
        self.assertIsNone(local['server_queue_seconds'])
        self.assertTrue(sent['dispatched_to_server'])
        self.assertEqual(sent['server_queue_seconds'], .5)
        self.assertFalse(server)
