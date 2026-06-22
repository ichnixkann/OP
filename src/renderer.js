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
const resetSettings = $('resetSettings');
const closeSettings = $('closeSettings');
const settingsMessage = $('settingsMessage');

// Settings inputs
const wakeWordInput = $('wakeWordInput');
const sensitivityInput = $('sensitivityInput');
const sttModelInput = $('sttModelInput');
const sttLanguageInput = $('sttLanguageInput');
const ttsModelInput = $('ttsModelInput');
const ttsSpeakerInput = $('ttsSpeakerInput');
const llmModelInput = $('llmModelInput');
const llmMaxTokensInput = $('llmMaxTokensInput');
const llmTemperatureInput = $('llmTemperatureInput');
const silenceThresholdInput = $('silenceThresholdInput');
const silenceDurationInput = $('silenceDurationInput');
const minRecordingInput = $('minRecordingInput');
const maxRecordingInput = $('maxRecordingInput');
const deviceInput = $('deviceInput');
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
let reconnectTimer = null;
let currentState = 'idle';

// --- Debug logging (visible in UI + forwarded to main process console) ---
function debug(msg, level) {
  level = level || 'info';
  var time = new Date().toLocaleTimeString();
  var line = document.createElement('div');
  line.className = 'log-line log-' + level;
  line.textContent = '[' + time + '] ' + msg;
  if (debugContent) {
    debugContent.appendChild(line);
    // Keep log from growing unbounded.
    while (debugContent.childElementCount > 200) {
      debugContent.removeChild(debugContent.firstChild);
    }
    debugContent.scrollTop = debugContent.scrollHeight;
  }
  if (level === 'error') console.error(msg);
  else if (level === 'warn') console.warn(msg);
  else console.log(msg);
}

// --- UI helpers ---
function addMessage(role, text) {
  var welcome = conversation.querySelector('.welcome');
  if (welcome) welcome.remove();
  var el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  conversation.appendChild(el);
  conversation.scrollTop = conversation.scrollHeight;
}

function setState(state) {
  currentState = state;
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

function showSettingsMessage(text, type) {
  settingsMessage.textContent = text;
  settingsMessage.className = 'settings-message ' + (type || 'info');
  settingsMessage.classList.remove('hidden');
  setTimeout(function() {
    settingsMessage.classList.add('hidden');
  }, 4000);
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
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  var httpUrl = await getBackendUrl();
  var url = wsUrlFromHttp(httpUrl);
  debug('Connecting to WebSocket: ' + url);
  setWsStatus(false);

  try {
    ws = new WebSocket(url);
  } catch (err) {
    debug('WebSocket constructor failed: ' + err, 'error');
    scheduleReconnect();
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
    if (currentState !== 'idle') {
      setState('idle');
    }
    if (listening) {
      listening = false;
      pushBtn.classList.remove('active');
      pushBtn.textContent = 'Push to talk';
    }
    setHint('Disconnected. Reconnecting…');
    scheduleReconnect();
  };

  ws.onerror = function() {
    debug('WebSocket error', 'error');
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(function() {
    reconnectTimer = null;
    if (!wsConnected && initialized) {
      connect();
    }
  }, 2000);
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
  try {
    var binary = atob(base64Data);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    var float32 = new Float32Array(bytes.buffer);

    if (float32.length === 0) {
      debug('TTS audio empty, skipping playback', 'warn');
      return;
    }

    var buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
    buffer.copyToChannel(float32, 0);
    var node = audioCtx.createBufferSource();
    node.buffer = buffer;
    node.connect(audioCtx.destination);
    node.start();
    node.onended = function() { debug('TTS playback finished', 'ok'); };
  } catch (err) {
    debug('Audio playback error: ' + err, 'error');
  }
}

// --- Push to talk (THE user gesture that starts everything) ---
pushBtn.addEventListener('click', async function() {
  debug('Push to talk clicked');

  // First click: start mic + connect WebSocket, then enter listening mode
  if (!initialized) {
    debug('First click — initializing mic + WebSocket…');
    pushBtn.disabled = true;
    pushBtn.textContent = 'Starting…';

    var micOk = await startMic();
    if (!micOk) {
      pushBtn.disabled = false;
      pushBtn.textContent = 'Push to talk';
      return;
    }

    await connect();

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
  if (!settingsPanel.classList.contains('hidden')) {
    loadSettingsFromBackend();
  }
});
closeSettings.addEventListener('click', function() {
  settingsPanel.classList.add('hidden');
});

clearLogBtn.addEventListener('click', function() {
  if (debugContent) debugContent.innerHTML = '';
});

// --- Settings management ---
function populateSettingsUI(cfg) {
  wakeWordInput.value = cfg.wake_word || '';
  sensitivityInput.value = cfg.wake_word_sensitivity || 0.06;
  sttModelInput.value = cfg.stt_model || '';
  sttLanguageInput.value = cfg.stt_language || 'en';
  ttsModelInput.value = cfg.tts_model || '';
  ttsSpeakerInput.value = cfg.tts_speaker_embeddings || '';
  llmModelInput.value = cfg.llm_model || '';
  llmMaxTokensInput.value = cfg.llm_max_new_tokens || 256;
  llmTemperatureInput.value = cfg.llm_temperature || 0.7;
  silenceThresholdInput.value = cfg.silence_threshold || 0.02;
  silenceDurationInput.value = cfg.silence_duration_ms || 1200;
  minRecordingInput.value = cfg.min_recording_ms || 1000;
  maxRecordingInput.value = cfg.max_recording_ms || 15000;
  deviceInput.value = cfg.device || 'auto';
  enableCommands.checked = cfg.enable_command_execution !== false;
  enableLLM.checked = cfg.enable_llm_response !== false;
}

function gatherSettingsFromUI() {
  return {
    wake_word: wakeWordInput.value.trim() || 'computer',
    wake_word_sensitivity: parseFloat(sensitivityInput.value) || 0.06,
    stt_model: sttModelInput.value.trim(),
    stt_language: sttLanguageInput.value.trim() || 'en',
    tts_model: ttsModelInput.value.trim(),
    tts_speaker_embeddings: ttsSpeakerInput.value.trim(),
    llm_model: llmModelInput.value.trim(),
    llm_max_new_tokens: parseInt(llmMaxTokensInput.value, 10) || 256,
    llm_temperature: parseFloat(llmTemperatureInput.value) || 0.7,
    silence_threshold: parseFloat(silenceThresholdInput.value) || 0.02,
    silence_duration_ms: parseInt(silenceDurationInput.value, 10) || 1200,
    min_recording_ms: parseInt(minRecordingInput.value, 10) || 1000,
    max_recording_ms: parseInt(maxRecordingInput.value, 10) || 15000,
    device: deviceInput.value || 'auto',
    enable_command_execution: enableCommands.checked,
    enable_llm_response: enableLLM.checked,
  };
}

async function loadSettingsFromBackend() {
  var httpUrl = await getBackendUrl();
  try {
    var res = await fetch(httpUrl + '/config');
    if (res.ok) {
      var cfg = await res.json();
      populateSettingsUI(cfg);
      debug('Settings loaded from backend', 'ok');
    }
  } catch (err) {
    debug('Could not load settings: ' + err, 'warn');
  }
}

saveSettings.addEventListener('click', async function() {
  var patch = gatherSettingsFromUI();
  var httpUrl = await getBackendUrl();
  try {
    var res = await fetch(httpUrl + '/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    var data = await res.json();
    if (!res.ok) {
      var errMsg = (data.errors || []).join('; ') || 'Validation failed';
      showSettingsMessage(errMsg, 'error');
      debug('Settings validation failed: ' + errMsg, 'error');
      return;
    }
    if (data._restart_required) {
      var keys = (data._restart_keys || []).join(', ');
      showSettingsMessage('Saved. Restart required for: ' + keys, 'warn');
      addMessage('system', 'Settings saved. Restart the app for model/device changes to take effect (' + keys + ').');
    } else {
      showSettingsMessage('Settings saved successfully.', 'success');
      addMessage('system', 'Settings saved.');
    }
    debug('Settings saved', 'ok');
  } catch (err) {
    showSettingsMessage('Failed to save: ' + err.message, 'error');
    debug('Settings save failed: ' + err, 'error');
  }
});

resetSettings.addEventListener('click', async function() {
  var httpUrl = await getBackendUrl();
  try {
    var res = await fetch(httpUrl + '/config');
    if (res.ok) {
      // Populate with current defaults from backend.
      var cfg = await res.json();
      populateSettingsUI(cfg);
      showSettingsMessage('Fields reset to current saved values.', 'info');
    }
  } catch (err) {
    debug('Reset failed: ' + err, 'warn');
  }
});

// --- Init ---
(async function() {
  debug('Renderer loaded');
  var httpUrl = await getBackendUrl();
  debug('Backend URL: ' + httpUrl);
  try {
    var res = await fetch(httpUrl + '/config');
    if (res.ok) {
      var cfg = await res.json();
      populateSettingsUI(cfg);
      debug('Settings loaded from backend', 'ok');
    }
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
  setState('idle');
  setHint('Click "Push to talk" to begin.');
})();
