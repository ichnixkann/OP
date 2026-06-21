const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('assistant', {
  getBackendUrl: () => ipcRenderer.invoke('backend-url'),
});
