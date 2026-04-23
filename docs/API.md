# API Reference

Base URL: `http://<host>:<port>` (default `http://localhost:8111` for the
docker-compose setup).

Audio format for all uploads: **mono 16 kHz float32 little-endian PCM, no
container header**. Any byte length whose total is a multiple of 4 is
accepted; internally buffered into `chunk_size_sec` segments before
decoding.

## POST `/api/start`

Create a new streaming session.

**Query parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `language` | string | (auto-detect) | Optional forced language. One of the Qwen3-ASR supported languages (e.g. `Chinese`, `English`, `Japanese`). When set, the model emits text-only output without language-ID preamble. |

**Request body**: ignored.

**Response 200**
```json
{"session_id": "cb3a53d4bf1f42558b4fd2f65f3376b2"}
```

**Notes**: the session is reserved until either `/api/finish` is called or
`QASR_SESSION_TTL_SEC` seconds elapse with no activity.

---

## POST `/api/chunk?session_id=<id>`

Append audio to the session and run rolling decode. Returns the current
accumulated transcription.

**Query parameters**

| Name | Type | Required |
|------|------|----------|
| `session_id` | string | yes |

**Request headers**

| Name | Required | Value |
|------|----------|-------|
| `Content-Type` | recommended | `application/octet-stream` |

**Request body**: raw float32 little-endian PCM. Any length; the server
buffers sub-chunk-size audio and only decodes when it has accumulated
`chunk_size_sec` (default 1.0s) worth.

**Response 200**
```json
{"language": "Chinese", "text": "這是串流辨識的部分結果"}
```

`text` is the **accumulated** transcription of all audio received so far
in this session, not just the new chunk. Clients that want a delta should
diff against the previous response's `text`.

**Response 400**
* `{"detail":"invalid session_id"}` — session unknown or GC'd.
* `{"detail":"float32 bytes length not multiple of 4"}` — malformed body.

**Response 415**
* `{"detail":"expect application/octet-stream (f32 PCM)"}` — wrong
  Content-Type (empty is tolerated for convenience).

---

## POST `/api/finish?session_id=<id>`

Flush the final tail of audio (any samples buffered but shorter than one
full chunk), run a final decode, and destroy the session.

**Query parameters**

| Name | Type | Required |
|------|------|----------|
| `session_id` | string | yes |

**Request body**: ignored.

**Response 200**
```json
{"language": "Chinese", "text": "這是串流辨識完成後的最終結果"}
```

After a successful `/api/finish`, the session is deleted and any subsequent
calls with its id return 400.

---

## GET `/health`

Readiness probe. The response is always 200; inspect the `ready` field.

```json
{"ready": true, "sessions": 2}
```

`ready` is `false` during warmup (model loading and graph compilation take
roughly 60–120 seconds depending on GPU). Use as a Kubernetes readiness
probe with a sufficiently long `initialDelaySeconds`.

---

## GET `/api/sessions`

Introspection. Returns currently-live sessions (after an implicit GC of
idle ones).

```json
{"count": 2, "ids": ["cb3a53d4...", "48a12e60..."]}
```

Not intended for load-bearing client logic; exposed for debugging and
observability.

---

## GET `/`

Returns service metadata:

```json
{
  "service": "qasr-mt",
  "model": "Qwen/Qwen3-ASR-0.6B",
  "sessions": 0,
  "chunk_size_sec": 1.0
}
```

---

## Client patterns

### Naïve upload (whole utterance at once)

```bash
curl -X POST http://localhost:8111/api/start
# → {"session_id":"abc..."}
curl -X POST "http://localhost:8111/api/chunk?session_id=abc..." \
  -H "Content-Type: application/octet-stream" \
  --data-binary @utterance.f32pcm
curl -X POST "http://localhost:8111/api/finish?session_id=abc..."
# → {"language":"Chinese","text":"..."}
```

### Streaming (push chunks as they are recorded)

Push chunks as small as 20 ms or as large as you like; the server buffers
until `chunk_size_sec` has accumulated, then emits an updated transcription.

```python
sid = (await session.post(url_start)).json()["session_id"]
while still_recording:
    chunk = microphone.read(chunk_bytes)  # f32 PCM, any size
    resp = await session.post(f"{url_chunk}?session_id={sid}", data=chunk)
    partial = resp.json()["text"]
    ui.display(partial)
final = (await session.post(f"{url_finish}?session_id={sid}")).json()
```

See [`examples/python_client.py`](../examples/python_client.py) for a
working script.
