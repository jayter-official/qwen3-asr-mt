# Why this server exists

A diagnostic story. What started as "the speech input is acting weird" turned
into "we've been leaking one user's audio into another user's transcript."

## Symptoms

Users of a self-hosted Qwen3-ASR streaming deployment — a speech-input
client talking to an instance of `qwen-asr-demo-streaming` behind a reverse
proxy — began reporting three distinct issues over several days:

1. Occasional raw tokens visible in results: `language Chinese<asr_text>[transcription]`
2. Short utterances returning bizarre phrases that the user never spoke
   (training-data hallucinations on near-silent segments).
3. Seeing fragments of **someone else's transcription** appear in the partial
   panel.

(3) is the one that matters.

## First (wrong) hypothesis

The server's access log showed every request coming from a single IP, so the
initial reading was "it's all one user — client-side session leak plus some
LLM hallucination on short audio." Indeed both of those things were real:

* The client's streaming thread was not being reliably aborted when a new
  recording started, so old partials could still paint onto the panel.
* Qwen3-ASR, being an LLM-based ASR, will confabulate plausible-sounding
  phrases on silent or extremely short segments — the same class of failure
  Whisper is notorious for (`"Thanks for watching!"` and similar YouTube-trained
  artifacts appearing on silent audio). Real words, never spoken.

This framing made the fix look modest: tweak the client, accept some edge-case
hallucination, move on.

## The correction

One observation reframed the entire threat model:

> The single source IP the ASR server saw was the reverse proxy, not a
> single user.

Every downstream client had been invisible behind the proxy — what looked
like "one user occasionally seeing weird output" was in fact multiple
concurrent users sharing a server that was never designed for concurrency.

## The real root cause

Reading the upstream source made the architecture problem explicit. From
`qwen_asr/inference/qwen3_asr.py`:

```python
def streaming_transcribe(self, pcm16k, state):
    """
    Notes:
        - vLLM backend only.
        - No timestamps.
        - Single stream only (no batching).
    """
```

"Single stream only" is not a suggestion — it is a statement about what the
code path can handle. The demo server (`qwen_asr/cli/demo_streaming.py`) is a
Flask app with `threaded=True` (the default), which happily runs multiple
`streaming_transcribe` calls concurrently across threads, on top of a
**shared `vllm.LLM` offline engine**, a **module-level tokenizer**, and
module-level sampling params.

Under concurrent load this produces three independent failure modes:

1. **Tokenizer race on rolling decode.** Each step encodes the accumulated
   text, drops the last K tokens, decodes the prefix, and prepends it to the
   next prompt. When two threads do this concurrently on the same tokenizer,
   intermediate state can cross.
2. **vLLM offline `LLM.generate()` not designed for concurrent callers.**
   Shared request bookkeeping inside the engine can produce output that gets
   attributed to the wrong caller.
3. **Raw LID prefix leaks at boundaries.** The rolling-decode prefix is the
   tokenized form of `_raw_decoded`, which contains `language X<asr_text>` at
   its head. On malformed re-entries that prefix ends up in the next prompt,
   and `parse_asr_output` (which expects exactly one such prefix) can miss
   the nested case. The client sees `language Chinese<asr_text>...` verbatim
   in the response body.

All three are compatible with what users reported. (3) was directly visible
in saved logs. (1) and (2) explain why one user's text would appear in
another's partial panel: the content came from the *other* concurrently-running
decode whose tokens or bookkeeping bled into this request.

## Why not just add a lock?

Wrapping `streaming_transcribe` in a global lock would be correct but
throughput-fatal: it serialises all users through one GPU forward pass at a
time, erasing vLLM's entire reason for existing.

Wrapping only the tokenizer call is brittle (we'd have to rely on reading
the internals of the library to know what is and isn't shared).

Forking `qwen_asr` is a maintenance trap.

## Why not the official alternatives?

Surveyed in April 2026:

| Option | Why not |
|---|---|
| `vllm serve ... /v1/audio/transcriptions` | One-shot batch only. No partial/streaming UX. |
| `vllm serve ... /v1/realtime` WebSocket | Exists but [vllm#35767](https://github.com/vllm-project/vllm/issues/35767) lists five bugs: LID prefix leaks, hard-coded 5 s segments, no cross-segment context, no rollback, `input_stream` accepted but never read. |
| Community fix PR [vllm#35894](https://github.com/vllm-project/vllm/pull/35894) | Draft status, inactive 36 days, author themselves raise architectural concerns in [vllm#35908](https://github.com/vllm-project/vllm/issues/35908). Cherry-picking it requires rebuilding vLLM from source — longer than writing a proper server from scratch. |

## The insight that made the solution short

vLLM has **two** Python APIs:

* `vllm.LLM` — synchronous, offline batch. The demo uses this. Not designed
  for concurrent access.
* `vllm.AsyncLLMEngine` — asynchronous, fully concurrent-safe. Every
  `engine.generate(..., request_id=X)` call gets its own scheduler slot, KV
  cache, and request-level isolation. This is what `vllm serve` uses under
  the hood.

The fix is to keep the Qwen3-ASR SDK's rolling-decode logic (it works and is
model-specific, not worth reinventing) but swap the engine underneath:

```python
# Before (demo_streaming.py, unsafe):
outputs = self.model.generate([inp], sampling_params, use_tqdm=False)
gen_text = outputs[0].outputs[0].text

# After (qasr-mt, safe):
async for out in engine.generate(
    prompt=inp,
    sampling_params=sampling_params,
    request_id=f"{session_id}:c{chunk_id}",
):
    final = out
gen_text = final.outputs[0].text
```

Plus per-session state (so no two chunks in the same session re-enter the
decode), plus a regex sanitiser on the way out (so even a broken
`parse_asr_output` cannot leak `<asr_text>` to the client), plus keeping the
HTTP API shape of the demo so no client needs to change.

About two hundred lines of code. See
[`qasr_server/server.py`](../qasr_server/server.py).

## What was verified

A reproducible concurrent test: two clients start at roughly the same time,
each sending a different 10-second slice of the same 60-second WAV. Before
the fix, partial panels would bleed. After the fix, session A and session B
returned completely disjoint transcriptions matching their respective audio
slices — no fragment of A's text ever appeared in B's response or vice
versa, across every intermediate `/api/chunk` call and both final
`/api/finish` results.

Zero cross-contamination across all chunks. Zero `language X<asr_text>`
strings in any response body. Per-chunk latency under concurrency is
indistinguishable from the single-session case (~0.5 s).

The test is shipped as `tests/smoke_concurrent.py` and can be run against
any deployment.

## Takeaway

The mistake was assuming the demo server was production-shaped. It is not,
and the library explicitly says so. The fix is not to patch the demo but to
acknowledge that anything shared across multiple users needs `AsyncLLMEngine`,
not `LLM`. Everything else — the rolling-decode maths, the HTTP surface, the
parsing — is reusable.
