# Voice Assistant

A cross-platform desktop voice assistant powered by HuggingFace speech-to-speech models. Listens for a configurable wake word, transcribes your speech with Distil-Whisper, processes the text with a local LLM (or executes whitelisted commands), and replies with synthesized speech via SpeechT5.

## Architecture

```
voice-assistant/
├── electron/         Electron main process, tray, mic permission handler
│   ├── main.js
│   └── preload.js
├── src/              Renderer UI (HTML/CSS/JS)
│   ├── index.html
│   ├── styles.css
│   └── renderer.js
├── backend/          Python FastAPI backend
│   ├── server.py     WebSocket protocol + state machine
│   ├── stt.py        Distil-Whisper speech-to-text
│   ├── tts.py        SpeechT5 text-to-speech
│   ├── llm.py        Local HuggingFace LLM (Qwen2.5-1.5B-Instruct)
│   ├── wakeword.py   openWakeWord wake-word detection
│   ├── commands.py   Whitelisted command execution
│   └── config.py     Config loader + device resolver
├── config.json       Runtime configuration
├── package.json      Electron + builder config
└── start.sh          Launcher (backend + electron)
```

## Pipeline

1. Renderer captures microphone audio (16 kHz mono float32) and streams it over WebSocket.
2. Backend runs openWakeWord on each chunk. When the wake word fires, it switches to `listening` state.
3. Audio is buffered until silence is detected (configurable threshold/duration) or `max_recording_ms` is reached.
4. Buffer is transcribed with Distil-Whisper → transcript sent to UI.
5. The command router checks for whitelisted intents (time, date, open app, web search). If matched, the canned response is used.
6. Otherwise the local LLM generates a response.
7. The response is synthesized with SpeechT5 and streamed back as PCM audio; the renderer plays it.
8. State returns to `idle` and the cycle repeats.

## Usage

1. Launch the app with `./start.sh` (or `npm start`).
2. A **"Click to start"** overlay appears. Click the **Start** button — this is required by Chromium's autoplay policy to enable microphone access and audio playback.
3. The backend loads AI models (first run downloads ~5 GB, subsequent runs are instant from cache).
4. Once the status shows "Ready", either:
   - **Click "Push to talk"** to start listening, speak, then click again to stop.
   - **Say the wake word** (any speech above the sensitivity threshold triggers listening).
5. Your speech is transcribed and shown in the "Last heard" panel and the conversation log.
6. The assistant responds with text + synthesized speech.
7. The **Debug log** at the bottom shows real-time status: WebSocket connection, mic status, audio chunk count, transcripts, and errors.

## Setup

### 1. System dependencies

- Python 3.10–3.12 (Python 3.13+ may lack torch wheels — see troubleshooting)
- Node.js 18+ and npm
- ffmpeg (for any audio format conversions)
- PortAudio (for `sounddevice`): `sudo apt install portaudio19-dev` on Debian/Ubuntu

### 2. Python backend

```bash
cd voice-assistant

# Recommended: create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# CPU-only torch (smaller download)
pip install -r backend/requirements.txt

# OR with CUDA (Linux, adjust for your CUDA version):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

### 3. Electron app

```bash
npm install
```

### 4. Run

```bash
./start.sh
```

Or run the pieces manually in two terminals:

```bash
# Terminal 1
python3 -m uvicorn backend.server:app --host 127.0.0.1 --port 8765

# Terminal 2
npm start
```

## Configuration

Edit `config.json` (or use the in-app Settings panel):

| Key | Default | Description |
|-----|---------|-------------|
| `wake_word` | `computer` | Wake word (mapped to closest openWakeWord preset) |
| `wake_word_sensitivity` | `0.06` | Detection threshold (0.01–0.5, lower = stricter). Used as RMS energy threshold for the fallback detector. |
| `stt_model` | `distil-whisper/distil-large-v3` | HuggingFace STT model |
| `stt_language` | `en` | Spoken language |
| `tts_model` | `microsoft/speecht5_tts` | HuggingFace TTS model |
| `tts_speaker_embeddings` | `Matthijs/cmu_xim2` | Speaker embedding dataset |
| `llm_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Local LLM for responses |
| `llm_max_new_tokens` | `256` | Max response length |
| `llm_temperature` | `0.7` | Sampling temperature |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `silence_threshold` | `0.01` | RMS below which audio counts as silence |
| `silence_duration_ms` | `800` | Silence needed to end an utterance |
| `max_recording_ms` | `15000` | Hard cap on a single utterance |
| `enable_command_execution` | `true` | Allow whitelisted command intents |
| `enable_llm_response` | `true` | Use the LLM for non-command responses |

## Activation modes

The assistant supports three ways to start listening:

1. **Wake word** (always-listening): Any speech above the sensitivity threshold triggers listening. With the default `0.06` RMS threshold, normal speech triggers it. Lower the value to make it stricter (only loud speech), raise it to make it more sensitive.
2. **Push to talk (hold)**: Press and hold the "Push to talk" button. Release when done speaking.
3. **Push to talk (click toggle)**: Click the button once to start listening, click again to stop.

Push-to-talk sends a `force_listen` message that bypasses wake word detection entirely and enters listening mode immediately.

## Voice commands

The whitelisted command router understands:

- "what time is it" / "current time"
- "what is the date" / "what day is it"
- "open <app or url>" / "launch <app>"
- "search for <query>" / "google <query>" / "look up <query>"

Anything else is sent to the LLM.

## First run

The first launch downloads model weights:

- Distil-Whisper large-v3: ~1.5 GB
- SpeechT5 + HiFiGAN + speaker embeddings: ~600 MB
- Qwen2.5-1.5B-Instruct: ~3 GB
- openWakeWord models: ~20 MB

Total ~5 GB. Models are cached in `~/.cache/huggingface/` and reused on subsequent runs.

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'` on Python 3.13/3.14**
PyTorch wheels lag new Python releases. Use Python 3.11 or 3.12 from your distro or pyenv:
```bash
pyenv install 3.12 && pyenv local 3.12
```

**Mic permission denied (macOS)**
System Settings → Privacy & Security → Microphone → allow Electron.

**No audio playback in renderer**
The renderer needs a user gesture to start the AudioContext. Click anywhere in the window on first launch.

**GPU out of memory**
Set `"device": "cpu"` in `config.json`, or use smaller models:
- STT: `distil-whisper/distil-small.en`
- LLM: `Qwen/Qwen2.5-0.5B-Instruct`

**openWakeWord not detecting the wake word**
The configured wake word is mapped to the closest preset (`computer` → `hey_jarvis`). For a custom wake word, train an openWakeWord model and load it directly in `backend/wakeword.py`.

## Security

- Command execution is whitelisted to safe intents only (no shell injection surface).
- The backend binds to `127.0.0.1` only — not exposed to the network.
- No telemetry, no cloud calls — all inference is local.

## License

MIT
