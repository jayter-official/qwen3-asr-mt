#!/bin/bash
# Minimal curl walkthrough of the qasr-mt streaming API.
#
# Usage:
#   ./curl_demo.sh path/to/utterance.wav [BASE_URL]
#
# Requires ffmpeg (to convert WAV → 16k mono f32 PCM). If you already have
# raw f32 PCM you can replace the ffmpeg line with the file path directly.
set -e

WAV="${1:-utterance.wav}"
BASE="${2:-http://localhost:8111}"

if [ ! -f "$WAV" ]; then
  echo "usage: $0 <wav-file> [base-url]" >&2
  exit 2
fi

TMP="$(mktemp -u).f32pcm"
trap 'rm -f "$TMP"' EXIT

echo "[1/3] converting $WAV → 16 kHz mono float32 PCM"
ffmpeg -v error -y -i "$WAV" -f f32le -ac 1 -ar 16000 "$TMP"

echo "[2/3] opening session"
SID=$(curl -fsS -X POST "$BASE/api/start" | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "      session_id=$SID"

echo "[3/3] uploading audio"
curl -fsS -X POST "$BASE/api/chunk?session_id=$SID" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary "@$TMP" | python3 -m json.tool

echo "      finalising"
curl -fsS -X POST "$BASE/api/finish?session_id=$SID" | python3 -m json.tool
