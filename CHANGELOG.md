# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-04-23

First release. Production-safe drop-in replacement for
`qwen-asr-demo-streaming` (upstream Flask demo).

### Added

- FastAPI server built on `vllm.AsyncLLMEngine` for true multi-tenant
  concurrent streaming transcription.
- Per-session `asyncio.Lock` prevents same-session chunk races.
- Unique per-call `request_id` (`{session}:c{chunk}`) gives vLLM's
  scheduler the isolation it needs across concurrent users.
- Secondary regex sanitiser strips any residual
  `language X<asr_text>` metadata that `parse_asr_output` might miss on
  rolling-decode boundary cases.
- HTTP API compatible with upstream demo (`/api/start`, `/api/chunk`,
  `/api/finish`): existing clients are drop-in compatible.
- `/health` readiness probe and `/api/sessions` introspection endpoint.
- Idle session GC (`QASR_SESSION_TTL_SEC`, default 600 s).
- Optional `language` query parameter on `/api/start` to force language.
- `Dockerfile` that layers on top of `qwenllm/qwen3-asr:latest` (no vLLM
  rebuild).
- `docker-compose.yml` for one-command deployment with GPU + HF cache
  volume.
- Documentation: `README.md`, `docs/ARCHITECTURE.md`, `docs/WHY.md`,
  `docs/API.md`, client notes for Rust, curl + Python examples.
- Concurrent smoke test (`tests/smoke_concurrent.py`) verifying no
  cross-session state leakage and no LID metadata leak.

### Verified

- Two-session concurrent streaming on RTX 3090 with
  `Qwen/Qwen3-ASR-0.6B`: per-chunk latency ~0.5 s, no cross-talk, no
  metadata leak.
- Cold start ~60–120 s (model download + graph compile).

### Not yet supported

- Timestamps (inherited limitation from Qwen3-ASR SDK streaming path).
- Forced aligner integration.
- Authentication / rate limiting (recommend adding an nginx/ingress layer
  in front of the server for production multi-user deployments).
- Benchmarks beyond 2 concurrent sessions (vLLM claims up to 128 with
  the 0.6B model on a single GPU; untested here).
