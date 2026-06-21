// Renderer: captures microphone audio, streams it to the FastAPI backend.
//
// KEY INSIGHT: The first click on "Push to talk" is the user gesture that
// Chromium requires to enable microphone access and audio playback.
// No overlay needed — the button itself is the gesture.

const STATE_LABELS = {
  idle: 'Idle',
  loading: 'Loading…',
  listening: 'Listening…',
  transcribing: 'Transcribing…',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
};

const $ = (id) => document.getElementById(id);
const conversation = $('conversation');
const statusDot = $('statusDot');
const statusText = $('statusText');
const meterFill = $('meterFill');
const hint = $('hint');
const liveTranscript = $('liveTranscript');
const muteBtn = $('muteBtn');
const pushBtn = $('pushBtn');
const stopBtn = $('stopBtn');
const settingsBtn = $('settingsBtn');
const settingsPanel = $('settingsPanel');
const saveSettings = $('saveSettings');
const closeSettings = $('closeSettings');
const wakeWordInput = $('wakeWordInput');
const sensitivityInput = $('sensitivityInput');
const enableCommands = $('enableCommands');
const enableLLM = $('enableLLM');
const wsStatus = $('wsStatus');
const micStatus = $('micStatus');
const debugContent = $('debugContent');
const clearLogBtn = $('clearLogBtn');

let ws = null;
let audioCtx = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let muted = false;
let targetSampleRate = 16000;
let micStarted = false;
let audioChunksSent = 0;
let wsConnected = false;
let listening = false;
let initialized = false;

// --- Debug logging (visible in UI + forwarded to main process console) ---
function debug(msg, level) {
  level = level || 'info';
  var time = new Date().toLocaleTimeString();
  var line = document.createElement('div');
  line.className = 'log-line log-' + level;
  line.textContent = '[' + time + '] ' + msg;
  if (debugContent) {
    debugContent.appendChild(line);
    debugContent.scrollTop = debugContent.scrollHeight;
  }
  if (level === 'error') console.error(msg);
  else if (level === 'warn') console.warn(msg);
  else console.log(msg);
}

// --- UI helpers ---
function addMessage(role, text) {
  // Remove welcome message if present
  var welcome = conversation.querySelector('.welcome');
  if (welcome) welcome.remove();
  var el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  conversation.appendChild(el);
  conversation.scrollTop = conversation.scrollHeight;
}

function setState(state) {
  statusDot.className = 'dot ' + state;
  statusText.textContent = STATE_LABELS[state] || state;
}

function setHint(text) { hint.textContent = text; }

function setLiveTranscript(text, cls) {
  liveTranscript.textContent = text;
  liveTranscript.className = 'live-transcript ' + (cls || '');
}

function setWsStatus(connected) {
  wsConnected = connected;
  wsStatus.textContent = connected ? 'WS: on' : 'WS: off';
  wsStatus.className = 'conn-badge ' + (connected ? 'connected' : 'disconnected');
}

function setMicStatus(on) {
  micStatus.textContent = on ? 'MIC: on' : 'MIC: off';
  micStatus.className = 'conn-badge ' + (on ? 'on' : 'off');
}

async function getBackendUrl() {
  if (window.assistant && window.assistant.getBackendUrl) {
    try {
      var url = await window.assistant.getBackendUrl();
      if (url) return url;
    } catch (err) {
      debug('IPC getBackendUrl failed: ' + err, 'warn');
    }
  }
  return 'http://127.0.0.1:8765';
}

function wsUrlFromHttp(httpUrl) {
  return httpUrl.replace(/^http/, 'ws') + '/ws';
}

// --- Audio capture (called from Push to Talk click = user gesture) ---
async function startMic() {
  if (micStarted) return true;
  debug('Requesting microphone access…');

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    debug('Microphone access granted', 'ok');
    setMicStatus(true);
  } catch (err) {
    debug('Microphone access FAILED: ' + err.name + ' — ' + err.message, 'error');
    addMessage('error', 'Microphone access failed: ' + err.message + '. Check system permissions.');
    return false;
  }

  // Create AudioContext INSIDE the user gesture — critical for Chromium.
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  debug('AudioContext created (state=' + audioCtx.state + ', sampleRate=' + audioCtx.sampleRate + ')');

  if (audioCtx.state === 'suspended') {
    debug('AudioContext suspended — resuming…');
    try {
      await audioCtx.resume();
      debug('AudioContext resumed (state=' + audioCtx.state + ')', 'ok');
    } catch (err) {
      debug('AudioContext resume failed: ' + err, 'error');
    }
  }

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);

  processorNode.onaudioprocess = function(e) {
    if (muted || !wsConnected) {
      meterFill.style.width = '0%';
      return;
    }
    var input = e.inputBuffer.getChannelData(0);
    var sum = 0;
    for (var i = 0; i < input.length; i++) sum += input[i] * input[i];
    var rms = Math.sqrt(sum / input.length);
    meterFill.style.width = Math.min(100, rms * 300) + '%';

    var resampled = downsampleToFloat32(input, audioCtx.sampleRate, targetSampleRate);
    sendJson({ type: 'audio', data: encodeBase64(resampled) });
    audioChunksSent++;
    if (audioChunksSent % 100 === 0) {
      debug('Audio flowing: ' + audioChunksSent + ' chunks, RMS=' + rms.toFixed(3), 'ok');
    }
  };

  micStarted = true;
  debug('Microphone pipeline active', 'ok');
  return true;
}

// --- WebSocket connection ---
async function connect() {
  var httpUrl = await getBackendUrl();
  var url = wsUrlFromHttp(httpUrl);
  debug('Connecting to WebSocket: ' + url);
  setWsStatus(false);

  try {
    ws = new WebSocket(url);
  } catch (err) {
    debug('WebSocket constructor failed: ' + err, 'error');
    return;
  }

  ws.onopen = function() {
    debug('WebSocket connected', 'ok');
    setWsStatus(true);
  };

  ws.onmessage = function(event) {
    var msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    switch (msg.type) {
      case 'ready':
        targetSampleRate = msg.sample_rate || 16000;
        debug('Backend ready (sample_rate=' + targetSampleRate + ')', 'ok');
        break;
      case 'status':
        setState(msg.state);
        if (msg.state === 'loading') {
          setHint('Loading AI models… (first run downloads ~5 GB)');
          setLiveTranscript('Loading models…', 'placeholder');
        } else if (msg.state === 'idle') {
          if (listening) {
            // Backend went idle but we think we're listening — sync state
            listening = false;
            pushBtn.classList.remove('active');
            pushBtn.textContent = 'Push to talk';
          }
          setHint('Ready. Click "Push to talk" to speak.');
          setLiveTranscript('—', 'placeholder');
        } else if (msg.state === 'listening') {
          setHint('Listening — speak now.');
          setLiveTranscript('Listening…', 'active');
        } else if (msg.state === 'transcribing') {
          setHint('Transcribing…');
          setLiveTranscript('Transcribing…', 'active');
        } else if (msg.state === 'thinking') {
          setHint('Thinking…');
        } else if (msg.state === 'speaking') {
          setHint('Speaking…');
        }
        break;
      case 'transcript':
        debug('Heard: "' + msg.text + '"', 'ok');
        setLiveTranscript(msg.text || '(empty)', '');
        addMessage('user', msg.text);
        break;
      case 'response':
        debug('Reply: "' + msg.text + '"', 'ok');
        addMessage('assistant', msg.text);
        break;
      case 'audio':
        debug('Playing TTS audio…');
        playAudio(msg.data, msg.sample_rate || 22050);
        break;
      case 'error':
        debug('Backend error: ' + msg.message, 'error');
        addMessage('error', msg.message);
        break;
      default:
        debug('Unknown message: ' + msg.type, 'warn');
    }
  };

  ws.onclose = function(event) {
    debug('WebSocket closed (code=' + event.code + ')', 'warn');
    setWsStatus(false);
    setState('idle');
    setHint('Disconnected. Reconnecting…');
    setTimeout(connect, 2000);
  };

  ws.onerror = function() {
    debug('WebSocket error', 'error');
  };
}

function sendJson(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

// --- Audio helpers ---
function downsampleToFloat32(buffer, inputRate, targetRate) {
  if (inputRate === targetRate) return buffer;
  var ratio = inputRate / targetRate;
  var newLen = Math.max(1, Math.floor(buffer.length / ratio));
  var out = new Float32Array(newLen);
  for (var i = 0; i < newLen; i++) {
    var idx = i * ratio;
    var lo = Math.floor(idx);
    var hi = Math.min(lo + 1, buffer.length - 1);
    var frac = idx - lo;
    out[i] = buffer[lo] * (1 - frac) + buffer[hi] * frac;
  }
  return out;
}

function encodeBase64(float32) {
  var bytes = new Uint8Array(float32.buffer);
  var binary = '';
  var chunk = 0x8000;
  for (var i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function playAudio(base64Data, sampleRate) {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(function() {});
  }
  var binary = atob(base64Data);
  var bytes = new Uint8Array(binary.length);
  for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  var float32 = new Float32Array(bytes.buffer);
  var buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
  buffer.copyToChannel(float32, 0);
  var node = audioCtx.createBufferSource();
  node.buffer = buffer;
  node.connect(audioCtx.destination);
  node.start();
  node.onended = function() { debug('TTS playback finished', 'ok'); };
}

// --- Push to talk (THE user gesture that starts everything) ---
pushBtn.addEventListener('click', async function() {
  debug('Push to talk clicked');

  // First click: start mic + connect WebSocket, then enter listening mode
  if (!initialized) {
    debug('First click — initializing mic + WebSocket…');
    pushBtn.disabled = true;
    pushBtn.textContent = 'Starting…';

    // 1. Start mic (MUST be in this user gesture)
    var micOk = await startMic();
    if (!micOk) {
      pushBtn.disabled = false;
      pushBtn.textContent = 'Push to talk';
      return;
    }

    // 2. Connect WebSocket
    await connect();

    // 3. Wait for WebSocket to be ready
    var waited = 0;
    while (!wsConnected && waited < 5000) {
      await new Promise(function(r) { setTimeout(r, 100); });
      waited += 100;
    }
    if (!wsConnected) {
      debug('WebSocket not connected after 5s — try anyway', 'warn');
    }

    initialized = true;
    pushBtn.disabled = false;
    debug('Initialization complete', 'ok');
  }

  // Toggle listening
  if (!listening) {
    listening = true;
    pushBtn.classList.add('active');
    pushBtn.textContent = 'Stop';
    debug('Listening ON — sending force_listen', 'ok');
    var sent = sendJson({ type: 'force_listen' });
    if (!sent) debug('force_listen send failed (WS not open)', 'error');
    setHint('Listening — speak now. Click again to stop.');
  } else {
    listening = false;
    pushBtn.classList.remove('active');
    pushBtn.textContent = 'Push to talk';
    debug('Listening OFF');
    sendJson({ type: 'stop' });
    setHint('Stopped. Click "Push to talk" to speak again.');
  }
});

// --- Other buttons ---
muteBtn.addEventListener('click', function() {
  muted = !muted;
  muteBtn.classList.toggle('active', muted);
  muteBtn.title = muted ? 'Unmute mic' : 'Mute mic';
  muteBtn.textContent = muted ? '🔇' : '🎙';
  debug('Mic ' + (muted ? 'muted' : 'unmuted'));
});

stopBtn.addEventListener('click', function() {
  sendJson({ type: 'stop' });
  if (listening) {
    listening = false;
    pushBtn.classList.remove('active');
    pushBtn.textContent = 'Push to talk';
  }
  debug('Stop clicked');
});

settingsBtn.addEventListener('click', function() {
  settingsPanel.classList.toggle('hidden');
});
closeSettings.addEventListener('click', function() {
  settingsPanel.classList.add('hidden');
});

clearLogBtn.addEventListener('click', function() {
  if (debugContent) debugContent.innerHTML = '';
});

saveSettings.addEventListener('click', async function() {
  var patch = {
    wake_word: wakeWordInput.value.trim() || 'computer',
    wake_word_sensitivity: parseFloat(sensitivityInput.value) || 0.06,
    enable_command_execution: enableCommands.checked,
    enable_llm_response: enableLLM.checked,
  };
  var httpUrl = await getBackendUrl();
  try {
    await fetch(httpUrl + '/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    addMessage('system', 'Settings saved.');
    debug('Settings saved', 'ok');
    settingsPanel.classList.add('hidden');
  } catch (err) {
    addMessage('error', 'Failed to save settings: ' + err.message);
    debug('Settings save failed: ' + err, 'error');
  }
});

// --- Init ---
(async function() {
  debug('Renderer loaded');
  var httpUrl = await getBackendUrl();
  debug('Backend URL: ' + httpUrl);
  try {
    var res = await fetch(httpUrl + '/config');
    var cfg = await res.json();
    wakeWordInput.value = cfg.wake_word || '';
    sensitivityInput.value = cfg.wake_word_sensitivity || 0.06;
    enableCommands.checked = !!cfg.enable_command_execution;
    enableLLM.checked = !!cfg.enable_llm_response;
    debug('Settings loaded from backend', 'ok');
  } catch (err) {
    debug('Could not load settings: ' + err, 'warn');
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    debug('getUserMedia NOT available!', 'error');
    addMessage('error', 'Microphone API not available in this Electron version.');
    pushBtn.disabled = true;
  } else {
    debug('getUserMedia available', 'ok');
  }
  // Do NOT connect WebSocket here — wait for Push to Talk click (user gesture).
  setState('idle');
  setHint('Click "Push to talk" to begin.');
})();
