"""Asynchronous DeepSeek judge using the frozen 181174 prompt and parser."""

import asyncio
import os
import threading
import time
from pathlib import Path
from statistics import mean, quantiles

import httpx

from .api_judge import API_URL
from .prepare_training import judge_protocol
from .stability import read_json, write_json


MODEL = 'deepseek-flash'
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FlashJudge:
    def __init__(self, key):
        if not key:
            raise RuntimeError('JUDGE_API_KEY is required in the Slurm environment')
        self.key = key
        self.protocol, self.identity = judge_protocol()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start(), self.loop).result()

    async def start(self):
        self.slots = asyncio.Semaphore(16)
        self.client = httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=None,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
            headers={'Authorization': 'Bearer ' + self.key},
        )

    def submit(self, question, final, gold, receipt):
        prompt = self.protocol['create_judge_prompt'](question, final, gold)

        async def recorded_verdict():
            row = await self.verdict(prompt)
            await asyncio.to_thread(write_json, receipt, row)
            return row

        return asyncio.run_coroutine_threadsafe(recorded_verdict(), self.loop)

    async def verdict(self, prompt):
        queued = time.monotonic()
        attempts = []
        verdict = served = None
        async with self.slots:
            started = time.monotonic()
            for number in range(4):
                attempt_start = time.monotonic()
                try:
                    async with asyncio.timeout(60):
                        response = await self.client.post(API_URL, json={
                            'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                            'temperature': .7, 'max_tokens': 4096, 'thinking': {'type': 'disabled'},
                            'stream': False,
                        })
                    status = response.status_code
                    if status == 200:
                        payload = response.json()
                        choice = payload['choices'][0]
                        parsed = self.protocol['parse_judge_response'](choice['message']['content'] or '')
                        if choice['finish_reason'] == 'stop' and not parsed['parse_error'] and parsed['correct'] is not None:
                            verdict, served = bool(parsed['correct']), payload.get('model')
                        else:
                            status = 'unresolved_verdict'
                except (TimeoutError, httpx.TransportError):
                    status = 'transport_error'
                except (ValueError, KeyError, IndexError, TypeError):
                    status = 'invalid_api_response'
                attempts.append({'status': status, 'seconds': time.monotonic() - attempt_start})
                if verdict is not None or (isinstance(status, int) and status not in RETRYABLE_STATUS):
                    break
                if number < 3:
                    await asyncio.sleep(2**number)
        ended = time.monotonic()
        return {'correct': verdict, 'served_model': served, 'queued_at': queued, 'started_at': started,
                'ended_at': ended, 'latency_seconds': ended - started,
                'queue_seconds': started - queued, 'retries': len(attempts) - 1, 'attempts': attempts}

    def close(self):
        asyncio.run_coroutine_threadsafe(self.client.aclose(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def summarize(rows):
    latencies = [row['latency_seconds'] for row in rows]
    return {'wall_seconds': max(row['ended_at'] for row in rows) - min(row['queued_at'] for row in rows),
            'mean_latency_seconds': mean(latencies),
            'p95_latency_seconds': quantiles(latencies, n=100, method='inclusive')[94],
            'retries': sum(row['retries'] for row in rows),
            'failures': sum(row['correct'] is None for row in rows),
            'served_models': sorted({row['served_model'] for row in rows if row['served_model']})}


def local_comparison():
    from vllm import LLM, SamplingParams

    path = Path(os.environ['OCCAM_LOCAL_JUDGE_INPUT'])
    data = read_json(path)
    protocol, _ = judge_protocol()
    llm = LLM(model=os.environ['OCCAM_LOCAL_JUDGE_MODEL'], tensor_parallel_size=2,
              max_model_len=32768, gpu_memory_utilization=.85, max_num_seqs=16,
              disable_custom_all_reduce=True)
    rows = []
    for index, record in enumerate(data):
        prompt = protocol['create_judge_prompt'](record['question'], record['final'], record['gold'])
        output = llm.chat([[{'role': 'user', 'content': prompt}]],
                          SamplingParams(temperature=.7, top_p=.8, top_k=20, max_tokens=4096, seed=2026111000+index),
                          chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)[0].outputs[0]
        parsed = protocol['parse_judge_response'](output.text)
        local = (bool(parsed['correct']) if output.finish_reason == 'stop' and
                 not parsed['parse_error'] and parsed['correct'] is not None else None)
        api = record['api_correct']
        rows.append({'local_correct': local, 'api_correct': api, 'termination': record['termination'],
                     'agree': None if local is None or api is None else local == api})
    write_json(path.with_name('LOCAL_JUDGE.json'), rows)


if __name__ == '__main__':
    local_comparison()
