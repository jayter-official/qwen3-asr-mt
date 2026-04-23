# Notes for Rust clients

The server's wire protocol is the same as the upstream
`qwen-asr-demo-streaming` Flask demo. If you already have a Rust client
talking to the demo, **you do not need to change anything** except the
endpoint URL.

If you are writing a new client, three sketch snippets using `ureq`:

```rust
// start
let resp: StartResp = ureq::post(&format!("{base}/api/start"))
    .send_bytes(&[])?
    .into_json()?;
let sid = resp.session_id;

// chunk
let mut bytes = Vec::with_capacity(samples.len() * 4);
for &s in samples {
    bytes.extend_from_slice(&s.to_le_bytes());    // f32 LE
}
let resp: AsrResp = ureq::post(&format!("{base}/api/chunk?session_id={sid}"))
    .set("Content-Type", "application/octet-stream")
    .send_bytes(&bytes)?
    .into_json()?;
println!("partial: {}", resp.text);

// finish
let resp: AsrResp = ureq::post(&format!("{base}/api/finish?session_id={sid}"))
    .send_bytes(&[])?
    .into_json()?;
println!("final: {}", resp.text);
```

With `#[derive(Deserialize)]` structs:

```rust
#[derive(Deserialize)]
struct StartResp { session_id: String }

#[derive(Deserialize)]
struct AsrResp {
    #[serde(default)] language: String,
    #[serde(default)] text: String,
}
```

## Practical tips

* **Resample on the client.** Send 16 kHz. The server will happily accept
  other buffer sizes but if you pass 48 kHz samples they'll be decoded as
  if they were 16 kHz (three times too fast → garbled transcription).
* **Chunk size is up to you.** Push 100 ms chunks for low-latency UX; the
  server buffers until it has `chunk_size_sec` (default 1 s) worth before
  running a decode. Sending smaller pieces just means the decode fires a
  bit later relative to your last chunk, not additional work.
* **The `text` field is cumulative**, not a delta. Overwrite your UI's
  transcription each time, don't append.
* **Handle thread lifetime.** If your UI lets the user start a new recording
  before the previous session's finish is in-flight, make sure the old
  streaming task is cancelled before you start the next `/api/start`. The
  server is safe against this (your old task's chunks will still be
  accepted into the old session), but your client's partial panel will
  show ghostly old text otherwise.
