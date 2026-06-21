const { app, BrowserWindow, ipcMain, session, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let tray = null;
let backendProcess = null;

const BACKEND_PORT = 8765;
const BACKEND_HOST = '127.0.0.1';
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

function loadConfig() {
  try {
    const fs = require('fs');
    const cfgPath = path.join(__dirname, '..', 'config.json');
    if (fs.existsSync(cfgPath)) {
      return JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    }
  } catch (err) {
    console.error('Failed to load config.json:', err);
  }
  return { backend_host: BACKEND_HOST, backend_port: BACKEND_PORT };
}

function startBackend() {
  const config = loadConfig();
  const projectRoot = path.join(__dirname, '..');
  const venvPython = path.join(projectRoot, '.venv', 'bin', 'python');
  const fs = require('fs');
  const candidates = [];
  if (fs.existsSync(venvPython)) candidates.push(venvPython);
  candidates.push('python3', 'python');

  const port = String(config.backend_port || BACKEND_PORT);
  const host = config.backend_host || BACKEND_HOST;

  // Check if backend is already running (e.g. started by start.sh).
  const http = require('http');
  const checkReq = http.request({ host, port, path: '/health', method: 'GET', timeout: 2000 }, (res) => {
    if (res.statusCode === 200) {
      console.log(`[backend] Already running on ${host}:${port}, not starting a new one.`);
      return;
    }
    spawnBackend();
  });
  checkReq.on('error', () => spawnBackend());
  checkReq.on('timeout', () => { checkReq.destroy(); spawnBackend(); });
  checkReq.end();

  function spawnBackend() {
    const trySpawn = (idx) => {
      if (idx >= candidates.length) {
        console.error('Could not start backend: no python found');
        return;
      }
      const py = candidates[idx];
      backendProcess = spawn(py, ['-m', 'uvicorn', 'backend.server:app', '--host', host, '--port', port], {
        cwd: projectRoot,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      backendProcess.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
      backendProcess.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
      backendProcess.on('error', () => trySpawn(idx + 1));
      backendProcess.on('exit', (code) => console.log(`Backend exited with code ${code}`));
    };
    trySpawn(0);
  }
}

function createTray() {
  // 16x16 transparent icon (no asset shipped by default).
  const icon = nativeImage.createEmpty();
  tray = new Tray(icon);
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show', click: () => mainWindow && mainWindow.show() },
    { label: 'Quit', click: () => app.quit() },
  ]);
  tray.setToolTip('Voice Assistant');
  tray.setContextMenu(contextMenu);
  tray.on('click', () => mainWindow && mainWindow.show());
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 500,
    height: 800,
    minWidth: 360,
    minHeight: 500,
    show: true,
    frame: true,
    title: 'Voice Assistant',
    backgroundColor: '#0f1117',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, '..', 'src', 'index.html'));
  mainWindow.on('close', (e) => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  // Auto-grant microphone permission requests from the renderer.
  session.defaultSession.setPermissionRequestHandler((wc, permission, cb) => {
    console.log(`[permission] request: ${permission}`);
    cb(permission === 'media');
  });
  // Also handle permission checks (Chromium may check before requesting).
  session.defaultSession.setPermissionCheckHandler((wc, permission, requestingOrigin) => {
    const allowed = permission === 'media';
    if (!allowed) console.log(`[permission] check denied: ${permission} from ${requestingOrigin}`);
    return allowed;
  });

  // Open DevTools in development for debugging.
  if (process.env.VOICE_ASSISTANT_DEV) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  // Forward renderer console messages to stdout for debugging.
  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    const levelStr = ['LOG', 'WARN', 'ERROR'][level] || 'LOG';
    console.log(`[renderer:${levelStr}] ${message} (${sourceId}:${line})`);
  });

  // Log any render process crashes.
  mainWindow.webContents.on('render-process-gone', (event, details) => {
    console.error(`[renderer:CRASH] ${JSON.stringify(details)}`);
  });
}

app.whenReady().then(() => {
  startBackend();
  createTray();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    // Keep tray alive; user quits via tray menu.
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
  else mainWindow && mainWindow.show();
});

app.on('before-quit', () => {
  app.isQuitting = true;
  if (backendProcess) {
    try { backendProcess.kill(); } catch (_) {}
  }
});

ipcMain.handle('backend-url', () => BACKEND_URL);
