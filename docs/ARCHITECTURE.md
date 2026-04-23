# Architecture

## Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Container                                  │
│                                                                         │
│   ┌─────────────┐      ┌────────────────────┐     ┌──────────────────┐  │
│   │  FastAPI    │      │  Session registry  │     │  AsyncLLMEngine  │  │
│   │  (async)    │◄────►│  dict[sid]→state   │     │  (shared, 1x)    │  │
│   │             │      │  ├─ asyncio.Lock   │     │                  │  │
│   │  /api/start │      │  │                 │     │  - scheduler     │  │
│   │  /api/chunk │      │  └─ per-session    │     │  - KV cache      │  │
│   │  /api/finish│      │     locks          │     │  - chunked       │  │
│   │             │      │                    │     │    prefill       │  │
│   └──────┬──────┘      └────────────────────┘     └────────▲─────────┘  │
│          │                                                  │            │
│          │  holds state.lock                                │            │
│          ▼                                                  │            │
│   ┌──────────────────────────────────────────────┐          │            │
│   │  rolling decode loop                         │          │            │
│   │   1. append wav → state.buffer               │          │            │
│   │   2. while buffer ≥ chunk_size_samples:      │          │            │
│   │        chunk = buffer[:chunk]                │          │            │
│   │        audio_accum += chunk                  │          │            │
│   │        prefix = rollback(state._raw_decoded) │          │            │
│   │        prompt = prompt_raw + prefix          │          │            │
│   │        gen = await engine.generate(          │──────────┘            │
│   │                prompt={prompt, audio},       │                       │
│   │                request_id=sid:chunk_id)     │                       │
│   │        _raw_decoded = prefix + gen           │                       │
│   │        lang, txt = parse_asr_output(...)     │                       │
│   │        state.text = sanitize(txt)            │                       │
│   │        state.chunk_id += 1                   │                       │
│   └──────────────────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Multi-tenancy invariants

Safety depends on two invariants being held together:

### Invariant 1: engine isolation by `request_id`

Every call to `engine.generate()` uses a unique `request_id` of the form
`{session_uuid}:c{chunk_id}`. vLLM's internal scheduler treats distinct
request_ids as fully separate requests — distinct KV cache pages,
separate logits buffers, separate output token streams. Two sessions'
requests can interleave inside one batch step but their outputs cannot
mix.

This is the contract `AsyncLLMEngine` is designed around. It is what
`vllm serve` depends on for `/v1/chat/completions` multi-tenancy, which
vLLM has been running in production at scale.

### Invariant 2: state isolation per session

A session's state — accumulated audio, committed text, chunk counter,
prompt — is a dataclass instance living in the `SESSIONS` dict. It is
only ever mutated inside its own `asyncio.Lock`. This matters for two
reasons:

1. A single session can receive multiple `/api/chunk` requests concurrently
   (if the client doesn't wait for one response before sending the next).
   Without the lock, two chunks could each trigger a decode that reads the
   same `_raw_decoded` and then both write incremented `chunk_id` back —
   classic read-modify-write race.
2. Idle GC can run while a chunk is in flight. The outer `SESSIONS_LOCK`
   covers insertion/deletion of the dict itself; per-session locks cover
   the state mutation.

These two locks never nest in a way that can deadlock: a chunk handler
first takes `SESSIONS_LOCK` briefly to resolve the session, releases it,
then takes `state.lock` for the decode. The GC path only touches
`SESSIONS_LOCK`.

## Why `AsyncLLMEngine` and not `vllm.LLM`

`vllm.LLM` is the synchronous offline-inference API:

```python
# vllm.LLM — synchronous, blocking
llm = LLM(model=...)
outputs = llm.generate(prompts, sampling_params)  # blocks until batch done
```

Internally it manages its own engine, its own scheduler, and its own request
IDs. Calling it from multiple threads simultaneously is not a supported
pattern. The engine may be reentrant for some operations and not for others;
relying on that is a brittle basis for a multi-tenant service.

`AsyncLLMEngine` is the async online-inference API:

```python
# vllm.AsyncLLMEngine — asynchronous, concurrent-safe by design
engine = AsyncLLMEngine.from_engine_args(args)
async for out in engine.generate(prompt, sampling_params, request_id=X):
    ...
```

It is explicitly designed for the `vllm serve` use case: many concurrent
requests sharing one engine instance, each tagged with a unique `request_id`.
The scheduler batches requests from different callers while keeping their
outputs separate.

Using `AsyncLLMEngine` inside our FastAPI app gives us the same guarantees
`vllm serve` provides, without needing to route through `vllm serve`'s
OpenAI-compatible wrapper (which currently has no good fit for stateful
streaming ASR — see [WHY.md](WHY.md)).

## Rolling-decode algorithm (ported from Qwen3-ASR SDK)

The SDK's `streaming_transcribe` performs this loop each time enough audio
has accumulated to form one chunk:

```
if chunk_id < unfixed_chunk_num:
    prefix = ""
else:
    # Drop last unfixed_token_num tokens from the previously-accumulated
    # decoded text, so the model can revise those tokens given new audio.
    token_ids = tokenizer.encode(_raw_decoded)
    prefix = tokenizer.decode(token_ids[: len(token_ids) - unfixed_token_num])

prompt = prompt_raw + prefix
gen_text = engine.generate(prompt, audio=audio_accum)
_raw_decoded = prefix + gen_text
language, text = parse_asr_output(_raw_decoded, user_language=force_language)
```

Three parameters control the trade-off between latency and boundary
accuracy:

* `chunk_size_sec` (default 1.0s) — how often a new segment is emitted.
  Smaller = more responsive partials, more GPU work.
* `unfixed_chunk_num` (default 4) — first N chunks decode with an empty
  prefix, letting the model commit its initial hypothesis without
  anchoring on premature text.
* `unfixed_token_num` (default 5) — how many trailing tokens are
  "unfixed" (subject to revision) when constructing the prefix for the
  next step. Larger = better boundary correction, more tokens regenerated.

These defaults match Qwen's own streaming SDK defaults and the upstream
demo; we do not change them.

## LID metadata sanitisation

Qwen3-ASR produces output in the format
`language {Lang}<asr_text>{transcription}`. The SDK ships
`parse_asr_output(raw)` which splits on `<asr_text>` and returns
`(language, text)`.

Edge cases where this can fail:

1. Malformed output with multiple `<asr_text>` markers.
2. Rolling-decode prefix that already contains `language X<asr_text>` from
   the previous `_raw_decoded`.
3. Truncated output where the marker is incomplete.

To make the client-facing response robust, we pass every result through
a secondary regex sanitiser:

```python
_LID_LEAK_RE = re.compile(r"(?:^|\s)language\s+[A-Za-z\-]+<asr_text>")

def _sanitize(text):
    while m := _LID_LEAK_RE.search(text):
        text = text[m.end():]
    if text.startswith("<asr_text>"):
        text = text[len("<asr_text>"):]
    return text.lstrip()
```

This is defence in depth; `parse_asr_output` handles the normal case, the
regex cleans up anything it missed. In practice with `AsyncLLMEngine` we
have not observed `parse_asr_output` miss a case, but the cost of the
regex is a microsecond per response and the alternative is silently shipping
metadata to the client.

## Deployment

The repo ships a `Dockerfile` that layers this server on top of the upstream
`qwenllm/qwen3-asr:latest` image. We do not re-bundle vLLM or CUDA — the base
image has them. We add only FastAPI, uvicorn, and the application code.

The `docker-compose.yml` wires in:
* GPU access via the NVIDIA container toolkit
* A mounted HuggingFace cache so the model doesn't re-download on every
  container recreate
* Restart policy `unless-stopped`
* A `shm_size` of 2 GB (vLLM benefits from generous shared memory for
  inter-process tensor passing)

For production placement on a GPU shared with other services (e.g. ComfyUI,
other models), tune `QASR_GPU_MEM_UTIL` conservatively. On RTX 3090 (24 GB)
with ComfyUI already consuming ~7 GB, 0.35 leaves the 0.6B model comfortably
within budget.
