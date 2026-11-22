#pragma once

#include <string_view>

namespace ingine::serve {

inline constexpr std::string_view kWebUiHtml = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BlackInk — Inkling-2-Mini 128K Studio</title>
  <style>
    :root {
      --bg: #08090c;
      --panel: #101218;
      --panel-hover: #171a22;
      --border: #222634;
      --accent: #7c5cff;
      --accent-glow: rgba(124, 92, 255, 0.25);
      --text: #e6e8f0;
      --muted: #8a90a6;
      --user-bg: #1c1f2e;
      --bot-bg: #12151e;
      --think-bg: #0c0e14;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      overflow: hidden;
    }
    /* Sidebar */
    .sidebar {
      width: 280px;
      background: var(--panel);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 16px;
      gap: 16px;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .brand-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .ink-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px var(--accent);
    }
    .badge {
      font-size: 0.7rem;
      background: rgba(124, 92, 255, 0.15);
      color: #b4a2ff;
      border: 1px solid rgba(124, 92, 255, 0.35);
      padding: 2px 7px;
      border-radius: 999px;
      font-weight: 600;
    }
    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 14px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.875rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .btn:hover { filter: brightness(1.1); box-shadow: 0 0 16px var(--accent-glow); }
    .btn-secondary {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
    }
    .btn-secondary:hover { background: var(--panel-hover); box-shadow: none; }
    .btn-danger { background: var(--danger); }
    .chat-list {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .chat-item {
      padding: 9px 12px;
      border-radius: 8px;
      font-size: 0.85rem;
      color: var(--muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border: 1px solid transparent;
      transition: all 0.12s;
    }
    .chat-item:hover { background: var(--panel-hover); color: var(--text); }
    .chat-item.active {
      background: rgba(124, 92, 255, 0.12);
      color: var(--text);
      border-color: rgba(124, 92, 255, 0.35);
    }
    .chat-item-title {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 195px;
    }
    .chat-del {
      opacity: 0;
      background: none;
      border: none;
      color: var(--muted);
      cursor: pointer;
      font-size: 0.9rem;
    }
    .chat-item:hover .chat-del { opacity: 1; }
    .chat-del:hover { color: var(--danger); }

    /* Controls section in sidebar */
    .controls {
      border-top: 1px solid var(--border);
      padding-top: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      font-size: 0.8rem;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .control-label {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-weight: 500;
    }
    select, input[type="number"], textarea {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 0.82rem;
      outline: none;
    }
    select:focus, input:focus, textarea:focus { border-color: var(--accent); }
    input[type="range"] { accent-color: var(--accent); width: 100%; }
    .status-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      color: var(--muted);
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
    }

    /* Main chat area */
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: radial-gradient(circle at top right, rgba(124, 92, 255, 0.06), transparent 45%);
    }
    .header {
      padding: 14px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(16, 18, 24, 0.75);
      backdrop-filter: blur(8px);
    }
    .header-info {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .metrics {
      display: flex;
      gap: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      color: #b4a2ff;
    }
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .empty-state {
      margin: auto;
      text-align: center;
      max-width: 460px;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .empty-state h2 { color: var(--text); font-size: 1.4rem; }
    .msg {
      max-width: 840px;
      width: 100%;
      margin: 0 auto;
      padding: 16px 20px;
      border-radius: 12px;
      border: 1px solid var(--border);
      line-height: 1.6;
      font-size: 0.93rem;
    }
    .msg.user {
      background: var(--user-bg);
      border-color: rgba(124, 92, 255, 0.28);
    }
    .msg.assistant {
      background: var(--bot-bg);
    }
    .msg-role {
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }
    .think-box {
      background: var(--think-bg);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      margin-bottom: 12px;
      overflow: hidden;
    }
    .think-summary {
      padding: 8px 12px;
      font-size: 0.8rem;
      color: #b4a2ff;
      cursor: pointer;
      user-select: none;
      font-weight: 600;
    }
    .think-content {
      padding: 10px 14px;
      font-size: 0.84rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      white-space: pre-wrap;
      font-family: 'JetBrains Mono', monospace;
    }
    .msg-content {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg-content pre {
      background: #07080b;
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 0 0 8px 8px;
      overflow-x: auto;
      margin: 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.84rem;
    }
    .msg-content code {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255,255,255,0.07);
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 0.85em;
    }
    .msg-content pre code { background: none; padding: 0; }
    .code-block {
      margin: 12px 0;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #07080b;
    }
    .code-block pre { border: none; }
    .code-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 12px;
      background: #141722;
      border-bottom: 1px solid var(--border);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      color: var(--muted);
    }
    .code-lang {
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
      color: #b4a2ff;
    }
    .code-actions {
      display: flex;
      gap: 6px;
    }
    .code-btn {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 3px 9px;
      font-size: 0.73rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.12s;
    }
    .code-btn:hover {
      background: var(--panel-hover);
      border-color: var(--accent);
    }
    .code-btn.preview-btn {
      background: rgba(124, 92, 255, 0.2);
      border-color: rgba(124, 92, 255, 0.45);
      color: #d8ceff;
    }
    .code-btn.preview-btn:hover {
      background: var(--accent);
      color: #fff;
    }

    /* Live Preview Drawer (Artifacts / Canvas style) */
    .preview-pane {
      width: 48%;
      min-width: 360px;
      max-width: 900px;
      height: 100vh;
      background: var(--panel);
      border-left: 1px solid var(--border);
      display: none;
      flex-direction: column;
      flex-shrink: 0;
    }
    .preview-pane.open {
      display: flex;
    }
    .preview-header {
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #141722;
    }
    .preview-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text);
    }
    .preview-controls {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .preview-frame-wrap {
      flex: 1;
      background: #ffffff;
      position: relative;
      overflow: hidden;
    }
    .preview-frame {
      width: 100%;
      height: 100%;
      border: none;
      display: block;
      background: #ffffff;
    }

    /* Composer */
    .composer-wrap {
      padding: 16px 24px 22px;
      border-top: 1px solid var(--border);
      background: var(--panel);
    }
    .composer {
      max-width: 840px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    .composer textarea {
      flex: 1;
      min-height: 52px;
      max-height: 200px;
      padding: 12px 14px;
      font-size: 0.93rem;
      border-radius: 10px;
      resize: vertical;
      font-family: inherit;
    }
    @media (max-width: 768px) {
      .sidebar { display: none; }
    }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-title">
        <span class="ink-dot"></span>
        <span>BlackInk</span>
      </div>
      <span class="badge">NVFP4 · 96K Q4</span>
    </div>

    <button class="btn" id="newChatBtn">+ New Conversation</button>

    <div class="chat-list" id="chatList"></div>

    <div class="controls">
      <div class="control-group">
        <label class="control-label">
          <span>Reasoning Effort</span>
          <span id="effortBadge">medium</span>
        </label>
        <select id="effortSelect">
          <option value="none">none (direct answer)</option>
          <option value="minimal">minimal (0.1)</option>
          <option value="low">low (0.2)</option>
          <option value="medium" selected>medium (0.7)</option>
          <option value="high">high (0.9)</option>
          <option value="xhigh">xhigh (0.99)</option>
        </select>
      </div>

      <div class="control-group">
        <label class="control-label">
          <span>Temperature</span>
          <span id="tempVal">0.70</span>
        </label>
        <input type="range" id="tempRange" min="0" max="1.5" step="0.05" value="0.7">
      </div>

      <div class="control-group">
        <label class="control-label">
          <span>Max New Tokens</span>
        </label>
        <input type="number" id="maxTokensInput" value="32768" min="64" max="98304" step="256">
      </div>

      <div class="control-group">
        <label class="control-label"><span>System Prompt</span></label>
        <textarea id="sysPrompt" rows="2" placeholder="Optional system instructions..."></textarea>
      </div>
    </div>

    <div class="status-bar">
      <span class="status-pill">
        <span class="status-dot" id="healthDot"></span>
        <span id="modelLabel">connecting...</span>
      </span>
      <span id="ctxLabel">128K</span>
    </div>
  </aside>

  <main class="main">
    <header class="header">
      <div class="header-info">
        <strong id="activeTitle">New Conversation</strong>
      </div>
      <div class="metrics">
        <span id="metricTps">-- tok/s</span>
        <span id="metricTokens">0 tokens</span>
      </div>
    </header>

    <section class="messages" id="messages">
      <div class="empty-state" id="emptyState">
        <h2>BlackInk Studio</h2>
        <p>Inkling-2-Mini Abliterated running natively on Blackwell Tensor Cores (Full-NVFP4 weights + 128K Q4 KV Cache).</p>
      </div>
    </section>

    <footer class="composer-wrap">
      <div class="composer">
        <textarea id="promptInput" placeholder="Ask anything... (Enter to send, Shift+Enter for newline)"></textarea>
        <button class="btn" id="sendBtn">Send</button>
        <button class="btn btn-danger" id="stopBtn" style="display:none;">Stop</button>
      </div>
    </footer>
  </main>

  <aside class="preview-pane" id="previewPane">
    <div class="preview-header">
      <div class="preview-title">
        <span class="ink-dot"></span>
        <span id="previewTitleLabel">HTML Live Preview</span>
      </div>
      <div class="preview-controls">
        <button class="code-btn" id="previewReloadBtn" title="Reload preview">↻ Reload</button>
        <button class="code-btn" id="previewNewTabBtn" title="Open in new tab">⤢ New Tab</button>
        <button class="code-btn" id="previewDownloadBtn" title="Download HTML">⬇ Download</button>
        <button class="code-btn" id="previewCloseBtn" title="Close preview">✕</button>
      </div>
    </div>
    <div class="preview-frame-wrap">
      <iframe id="previewFrame" class="preview-frame" sandbox="allow-scripts allow-forms allow-modals allow-popups"></iframe>
    </div>
  </aside>

  <script>
    const STORAGE_KEY = 'blackink_chats_v1';
    let chats = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    let activeId = chats[0]?.id || null;
    let modelId = 'inkling-2-mini-abliterated';
    let abortCtrl = null;

    const elMessages = document.getElementById('messages');
    const elChatList = document.getElementById('chatList');
    const elPrompt = document.getElementById('promptInput');
    const elSend = document.getElementById('sendBtn');
    const elStop = document.getElementById('stopBtn');
    const elEffort = document.getElementById('effortSelect');
    const elEffortBadge = document.getElementById('effortBadge');
    const elTemp = document.getElementById('tempRange');
    const elTempVal = document.getElementById('tempVal');
    const elMaxTokens = document.getElementById('maxTokensInput');
    const elSysPrompt = document.getElementById('sysPrompt');
    const elMetricTps = document.getElementById('metricTps');
    const elMetricTokens = document.getElementById('metricTokens');

    elEffort.addEventListener('change', () => elEffortBadge.textContent = elEffort.value);
    elTemp.addEventListener('input', () => elTempVal.textContent = Number(elTemp.value).toFixed(2));

    async function checkHealth() {
      try {
        const r = await fetch('/v1/models');
        const data = await r.json();
        const card = data?.data?.[0];
        if (card) {
          modelId = card.id;
          document.getElementById('modelLabel').textContent = card.id;
          document.getElementById('ctxLabel').textContent = Math.round((card.max_model_len || 131072) / 1024) + 'K';
          document.getElementById('healthDot').style.background = '#22c55e';
        }
      } catch (e) {
        document.getElementById('modelLabel').textContent = 'offline';
        document.getElementById('healthDot').style.background = '#ef4444';
      }
    }
    checkHealth();
    setInterval(checkHealth, 10000);

    function saveChats() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    }

    function createChat() {
      const c = { id: 'chat_' + Date.now(), title: 'New Conversation', messages: [] };
      chats.unshift(c);
      activeId = c.id;
      saveChats();
      renderSidebar();
      renderMessages();
    }

    function getActiveChat() {
      let c = chats.find(x => x.id === activeId);
      if (!c) {
        createChat();
        c = chats[0];
      }
      return c;
    }

    function renderSidebar() {
      elChatList.innerHTML = '';
      chats.forEach(c => {
        const div = document.createElement('div');
        div.className = 'chat-item' + (c.id === activeId ? ' active' : '');
        const title = document.createElement('span');
        title.className = 'chat-item-title';
        title.textContent = c.title;
        const del = document.createElement('button');
        del.className = 'chat-del';
        del.textContent = '×';
        del.onclick = (e) => {
          e.stopPropagation();
          chats = chats.filter(x => x.id !== c.id);
          if (activeId === c.id) activeId = chats[0]?.id || null;
          saveChats();
          renderSidebar();
          renderMessages();
        };
        div.appendChild(title);
        div.appendChild(del);
        div.onclick = () => {
          activeId = c.id;
          renderSidebar();
          renderMessages();
        };
        elChatList.appendChild(div);
      });
    }

    function escapeHtml(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    const codeStore = [];
    let currentPreviewCode = '';
    const elPreviewPane = document.getElementById('previewPane');
    const elPreviewFrame = document.getElementById('previewFrame');

    function isPreviewableCode(lang, code) {
      const l = (lang || '').toLowerCase();
      if (l === 'html' || l === 'htm' || l === 'svg' || l === 'xml') return true;
      const t = code.trim();
      return /^<!doctype\s+html/i.test(t) || /^<html[\s>]/i.test(t) || /^<svg[\s>]/i.test(t) ||
             (/<(div|section|main|canvas|style|script|body|head|h1|table|form)\b/i.test(t) && /<\/(div|section|main|canvas|style|script|body|html|table|form)>/i.test(t));
    }

    window.openHtmlPreview = function(idx) {
      const code = codeStore[idx];
      if (typeof code !== 'string') return;
      currentPreviewCode = code;
      elPreviewPane.classList.add('open');
      elPreviewFrame.srcdoc = code;
    };

    window.openHtmlNewTab = function(idx) {
      const code = typeof idx === 'number' ? codeStore[idx] : currentPreviewCode;
      if (!code) return;
      const blob = new Blob([code], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
    };

    window.downloadHtmlCode = function(idx) {
      const code = typeof idx === 'number' ? codeStore[idx] : currentPreviewCode;
      if (!code) return;
      const blob = new Blob([code], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'blackink-preview-' + Date.now() + '.html';
      a.click();
      URL.revokeObjectURL(url);
    };

    window.copyCodeBlock = function(idx, btn) {
      const code = codeStore[idx];
      if (typeof code !== 'string') return;
      navigator.clipboard.writeText(code).then(() => {
        const old = btn.textContent;
        btn.textContent = '✓ Copied';
        setTimeout(() => { btn.textContent = old; }, 1400);
      });
    };

    document.getElementById('previewReloadBtn').onclick = () => {
      if (currentPreviewCode) elPreviewFrame.srcdoc = currentPreviewCode;
    };
    document.getElementById('previewNewTabBtn').onclick = () => window.openHtmlNewTab();
    document.getElementById('previewDownloadBtn').onclick = () => window.downloadHtmlCode();
    document.getElementById('previewCloseBtn').onclick = () => {
      elPreviewPane.classList.remove('open');
    };

    function renderCodeBlock(rawInner) {
      let lang = 'code';
      let code = rawInner;
      const firstNl = rawInner.indexOf('\n');
      if (firstNl !== -1) {
        const firstLine = rawInner.slice(0, firstNl).trim();
        if (/^[a-zA-Z0-9_+-]+$/.test(firstLine)) {
          lang = firstLine;
          code = rawInner.slice(firstNl + 1);
        }
      }
      const idx = codeStore.length;
      codeStore.push(code);
      const canPreview = isPreviewableCode(lang, code);
      const previewBtns = canPreview
        ? `<button class="code-btn preview-btn" onclick="openHtmlPreview(${idx})">▶ Preview</button>
           <button class="code-btn" onclick="openHtmlNewTab(${idx})" title="Open in new tab">⤢</button>`
        : '';
      return `<div class="code-block">
        <div class="code-header">
          <span class="code-lang">${escapeHtml(lang)}</span>
          <div class="code-actions">
            ${previewBtns}
            <button class="code-btn" onclick="copyCodeBlock(${idx}, this)">Copy</button>
          </div>
        </div>
        <pre><code>${escapeHtml(code)}</code></pre>
      </div>`;
    }

    function formatMarkdown(text) {
      if (!text) return '';
      // Handle closed ```...``` blocks and an unclosed trailing ```... block during streaming
      let normalized = text;
      const fenceCount = (text.match(/```/g) || []).length;
      if (fenceCount % 2 === 1) normalized += '\n```';
      const parts = normalized.split(/(```[\s\S]*?```)/g);
      return parts.map(part => {
        if (part.startsWith('```') && part.endsWith('```')) {
          return renderCodeBlock(part.slice(3, -3));
        }
        return escapeHtml(part).replace(/`([^`\n]+)`/g, '<code>$1</code>');
      }).join('');
    }

    function renderMessages() {
      codeStore.length = 0;
      const chat = chats.find(x => x.id === activeId);
      document.getElementById('activeTitle').textContent = chat?.title || 'New Conversation';
      elMessages.innerHTML = '';
      if (!chat || chat.messages.length === 0) {
        elMessages.innerHTML = `
          <div class="empty-state">
            <h2>BlackInk Studio</h2>
            <p>Inkling-2-Mini Abliterated running natively on Blackwell Tensor Cores (Full-NVFP4 weights + 128K Q4 KV Cache).</p>
          </div>`;
        return;
      }
      chat.messages.forEach(m => {
        const div = document.createElement('div');
        div.className = 'msg ' + m.role;
        let html = `<div class="msg-role"><span>${m.role}</span><span>${m.meta || ''}</span></div>`;
        if (m.reasoning) {
          html += `<details class="think-box" ${m.streaming && !m.content ? 'open' : ''}>
            <summary class="think-summary">Thinking Process</summary>
            <div class="think-content">${escapeHtml(m.reasoning)}</div>
          </details>`;
        }
        html += `<div class="msg-content">${formatMarkdown(m.content || '')}</div>`;
        div.innerHTML = html;
        elMessages.appendChild(div);
      });
      elMessages.scrollTop = elMessages.scrollHeight;
    }

    async function sendMessage() {
      const text = elPrompt.value.trim();
      if (!text || abortCtrl) return;
      elPrompt.value = '';
      const chat = getActiveChat();
      if (chat.messages.length === 0) {
        chat.title = text.slice(0, 38) + (text.length > 38 ? '...' : '');
      }
      chat.messages.push({ role: 'user', content: text });
      const botMsg = { role: 'assistant', content: '', reasoning: '', streaming: true };
      chat.messages.push(botMsg);
      renderSidebar();
      renderMessages();

      const apiMsgs = [];
      const sys = elSysPrompt.value.trim();
      if (sys) apiMsgs.push({ role: 'system', content: sys });
      chat.messages.slice(0, -1).forEach(m => {
        apiMsgs.push({ role: m.role, content: m.content });
      });

      const effort = elEffort.value;
      const body = {
        model: modelId,
        messages: apiMsgs,
        temperature: Number(elTemp.value),
        max_tokens: Number(elMaxTokens.value) || 32768,
        stream: true,
        stream_options: { include_usage: true }
      };
      if (effort === 'none') {
        body.enable_thinking = false;
      } else {
        body.reasoning_effort = effort;
      }

      abortCtrl = new AbortController();
      elSend.style.display = 'none';
      elStop.style.display = 'inline-flex';

      const t0 = performance.now();
      let tokenCount = 0;

      try {
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: abortCtrl.signal
        });
        if (!res.ok) {
          const errText = await res.text();
          botMsg.content = 'Error ' + res.status + ': ' + errText;
          botMsg.streaming = false;
          renderMessages();
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const payload = trimmed.slice(5).trim();
            if (payload === '[DONE]') continue;
            try {
              const parsed = JSON.parse(payload);
              const delta = parsed.choices?.[0]?.delta;
              if (delta) {
                if (delta.reasoning_content) {
                  botMsg.reasoning += delta.reasoning_content;
                  tokenCount++;
                }
                if (delta.content) {
                  botMsg.content += delta.content;
                  tokenCount++;
                }
                const elapsed = Math.max((performance.now() - t0) / 1000, 0.05);
                const tps = (tokenCount / elapsed).toFixed(1);
                elMetricTps.textContent = `${tps} tok/s`;
                elMetricTokens.textContent = `${tokenCount} chunks`;
                botMsg.meta = `${tps} tok/s`;
              }
              if (parsed.usage) {
                const comp = parsed.usage.completion_tokens || tokenCount;
                const elapsed = Math.max((performance.now() - t0) / 1000, 0.05);
                const tps = (comp / elapsed).toFixed(1);
                elMetricTps.textContent = `${tps} tok/s`;
                elMetricTokens.textContent = `${parsed.usage.total_tokens} total (${parsed.usage.prompt_tokens} prompt + ${comp} gen)`;
                botMsg.meta = `${comp} tok · ${tps} tok/s`;
              }
            } catch (_) {}
          }
          renderMessages();
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          botMsg.content += '\n[Connection error: ' + err.message + ']';
        }
      } finally {
        botMsg.streaming = false;
        abortCtrl = null;
        elSend.style.display = 'inline-flex';
        elStop.style.display = 'none';
        saveChats();
        renderMessages();
        // Auto-preview the latest HTML/SVG block when generation finishes if preview pane is open
        for (let i = codeStore.length - 1; i >= 0; --i) {
          if (isPreviewableCode('', codeStore[i])) {
            if (elPreviewPane.classList.contains('open')) {
              window.openHtmlPreview(i);
            }
            break;
          }
        }
      }
    }

    document.getElementById('newChatBtn').onclick = createChat;
    elSend.onclick = sendMessage;
    elStop.onclick = () => { if (abortCtrl) abortCtrl.abort(); };
    elPrompt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    if (chats.length === 0) createChat();
    else { renderSidebar(); renderMessages(); }
  </script>
</body>
</html>
)HTML";

}  // namespace ingine::serve
