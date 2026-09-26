/**
 * MULTI-AGENT CHAT — CLIENT APPLICATION
 */

document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // Application State
  const state = {
    conversations: [],
    currentConversationId: null,
    isGenerating: false,
    documents: [],
    memories: [],
    settings: {
      model: 'llama3.2:latest',
      temperature: 0.2,
      confidence: 0.65,
    }
  };

  const MAX_DOCUMENTS = 30;
  const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024; // 500 MB

  // DOM Elements
  const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarCollapseBtn: document.getElementById('sidebar-collapse-btn'),
    sidebarOpenBtn: document.getElementById('sidebar-open-btn'),
    sidebarBackdrop: document.getElementById('sidebar-backdrop'),
    newChatBtn: document.getElementById('new-chat-btn'),
    conversationList: document.getElementById('conversation-list'),
    
    // Chat & Messages
    currentChatTitle: document.getElementById('current-chat-title'),
    welcomeHero: document.getElementById('welcome-hero'),
    messagesList: document.getElementById('messages-list'),
    messagesContainer: document.getElementById('chat-messages-container'),
    agentThinkingCard: document.getElementById('agent-thinking-card'),
    thinkingStepDetail: document.getElementById('thinking-step-detail'),
    
    // Input & Plus Upload
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-message-btn'),
    inputPlusBtn: document.getElementById('input-plus-upload-btn'),
    directFileInput: document.getElementById('direct-file-input'),
    uploadStatusBar: document.getElementById('upload-status-bar'),
    uploadStatusText: document.getElementById('upload-status-text'),
    
    // Files Manager Header & Modal
    headerFilesBtn: document.getElementById('header-files-btn'),
    headerFileCount: document.getElementById('header-file-count'),
    filesModal: document.getElementById('files-modal'),
    closeFilesModalBtn: document.getElementById('close-files-modal-btn'),
    closeFilesBtn: document.getElementById('close-files-btn'),
    modalFileCountBadge: document.getElementById('modal-file-count-badge'),
    uploadedFilesList: document.getElementById('uploaded-files-list'),
    modalDropzone: document.getElementById('modal-dropzone'),
    modalFileInput: document.getElementById('modal-file-input'),

    // Long-Term Memory Header & Modal
    headerMemoryBtn: document.getElementById('header-memory-btn'),
    headerMemoryCount: document.getElementById('header-memory-count'),
    memoryModal: document.getElementById('memory-modal'),
    closeMemoryModalBtn: document.getElementById('close-memory-modal-btn'),
    closeMemoryBtn: document.getElementById('close-memory-btn'),
    modalMemoryCountBadge: document.getElementById('modal-memory-count-badge'),
    newMemoryInput: document.getElementById('new-memory-input'),
    newMemoryCategory: document.getElementById('new-memory-category'),
    addMemoryBtn: document.getElementById('add-memory-btn'),
    clearAllMemoriesBtn: document.getElementById('clear-all-memories-btn'),
    memoriesItemsList: document.getElementById('memories-items-list'),
    
    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    openSettingsBtn: document.getElementById('open-settings-btn'),
    closeSettingsModalBtn: document.getElementById('close-settings-modal-btn'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    cfgModelSelect: document.getElementById('cfg-model-select'),
    cfgTemperature: document.getElementById('cfg-temperature'),
    cfgTemperatureVal: document.getElementById('cfg-temperature-val'),
    cfgConfidence: document.getElementById('cfg-confidence'),
    cfgConfidenceVal: document.getElementById('cfg-confidence-val'),
    
    // Status
    statusModelLabel: document.getElementById('status-model-label'),
    llmStatusIndicator: document.getElementById('llm-status-indicator'),
  };

  // Configure Marked
  if (window.marked) {
    marked.setOptions({
      highlight: function(code, lang) {
        if (window.Prism && Prism.languages[lang]) {
          return Prism.highlight(code, Prism.languages[lang], lang);
        }
        return code;
      },
      breaks: true,
      gfm: true
    });
  }

  // --- INITIALIZATION ---
  initApp();

  function initApp() {
    loadSavedSettings();
    loadConversations();
    setupEventListeners();
    checkHealth();
    fetchDocumentsList();
    fetchMemoriesList();

    // Respect user's explicit preference: only open if user previously opened it
    const isSidebarSavedOpen = localStorage.getItem('sidebar_open') === 'true';
    if (elements.sidebar) {
      if (isSidebarSavedOpen && window.innerWidth > 768) {
        elements.sidebar.classList.remove('collapsed');
      } else {
        elements.sidebar.classList.add('collapsed');
        elements.sidebar.classList.remove('mobile-open');
        elements.sidebarBackdrop?.classList.add('hidden');
      }
    }

    if (state.conversations.length === 0) {
      createNewConversation();
    } else {
      selectConversation(state.conversations[0].id);
    }
  }

  // --- HEALTH CHECK ---
  async function checkHealth() {
    try {
      const response = await fetch('/health');
      if (response.ok) {
        const data = await response.json();
        const llmInfo = data.services?.llm;
        if (llmInfo && llmInfo.model_ready) {
          elements.statusModelLabel.textContent = llmInfo.default_model || state.settings.model;
          elements.llmStatusIndicator.querySelector('.status-dot').className = 'status-dot online';
        } else {
          elements.llmStatusIndicator.querySelector('.status-dot').className = 'status-dot';
        }
      }
    } catch (e) {
      elements.llmStatusIndicator.querySelector('.status-dot').className = 'status-dot';
    }
  }

  // --- DOCUMENT MANAGEMENT (Fetch, Upload, Delete) ---
  async function fetchDocumentsList() {
    try {
      const res = await fetch('/documents');
      if (res.ok) {
        const data = await res.json();
        state.documents = data.documents || [];
        renderFilesBadge();
        renderUploadedFilesList();
      }
    } catch (e) {
      console.warn('Failed to fetch documents:', e);
    }
  }

  function renderFilesBadge() {
    const count = state.documents.length;
    if (elements.headerFileCount) {
      elements.headerFileCount.textContent = count;
    }
    if (elements.modalFileCountBadge) {
      elements.modalFileCountBadge.textContent = `${count} / ${MAX_DOCUMENTS} files`;
    }
  }

  function renderUploadedFilesList() {
    if (!elements.uploadedFilesList) return;
    elements.uploadedFilesList.innerHTML = '';

    if (state.documents.length === 0) {
      elements.uploadedFilesList.innerHTML = `
        <div class="empty-files-hint">
          <p>No documents uploaded yet.</p>
          <span style="font-size: 11px; opacity: 0.7;">Upload up to 30 files (max 500 MB each) to enable RAG retrieval.</span>
        </div>
      `;
      return;
    }

    state.documents.forEach(doc => {
      const row = document.createElement('div');
      row.className = 'file-row-card';
      row.innerHTML = `
        <div class="file-row-left">
          <div class="file-row-meta">
            <span class="file-row-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
            <span class="file-row-sub">${formatBytes(doc.size || 0)} • ${doc.chunks || 0} chunks</span>
          </div>
        </div>
        <button class="btn-file-delete" data-filename="${escapeHtml(doc.filename)}" title="Delete document">
          <i data-lucide="trash-2"></i>
        </button>
      `;

      // Delete action
      const delBtn = row.querySelector('.btn-file-delete');
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const filename = delBtn.getAttribute('data-filename');
        if (confirm(`Delete "${filename}" and remove it from the knowledge base?`)) {
          await deleteDocumentFile(filename);
        }
      });

      elements.uploadedFilesList.appendChild(row);
    });

    if (window.lucide) window.lucide.createIcons();
  }

  async function uploadFileDirectly(file) {
    if (!file) return;

    // 1. Check max documents limit
    const isNew = !state.documents.some(d => d.filename === file.name);
    if (state.documents.length >= MAX_DOCUMENTS && isNew) {
      alert(`Storage limit reached: Maximum of ${MAX_DOCUMENTS} documents allowed. Please delete an existing file first.`);
      return;
    }

    // 2. Check 500 MB file size limit
    if (file.size > MAX_FILE_SIZE_BYTES) {
      alert(`File "${file.name}" (${formatBytes(file.size)}) exceeds the maximum allowed size of 500 MB.`);
      return;
    }

    elements.uploadStatusBar?.classList.remove('hidden');
    if (elements.uploadStatusText) {
      elements.uploadStatusText.textContent = `Indexing "${file.name}"...`;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/documents/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      await fetchDocumentsList();
      if (elements.uploadStatusText) {
        elements.uploadStatusText.textContent = `✓ Indexed "${file.name}"`;
      }
      setTimeout(() => {
        elements.uploadStatusBar?.classList.add('hidden');
      }, 2500);
    } catch (err) {
      alert(`Error uploading file: ${err.message}`);
      elements.uploadStatusBar?.classList.add('hidden');
    }
  }

  async function deleteDocumentFile(filename) {
    try {
      const res = await fetch(`/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        await fetchDocumentsList();
      } else {
        const err = await res.json();
        alert(`Failed to delete file: ${err.detail}`);
      }
    } catch (e) {
      alert(`Delete error: ${e.message}`);
    }
  }

  // --- LONG-TERM MEMORY MANAGEMENT ---
  async function fetchMemoriesList() {
    try {
      const res = await fetch('/memory');
      if (res.ok) {
        const data = await res.json();
        state.memories = data.memories || [];
        renderMemoryBadge();
        renderMemoriesList();
      }
    } catch (e) {
      console.warn('Failed to fetch memories:', e);
    }
  }

  function renderMemoryBadge() {
    const count = state.memories.length;
    if (elements.headerMemoryCount) {
      elements.headerMemoryCount.textContent = count;
    }
    if (elements.modalMemoryCountBadge) {
      elements.modalMemoryCountBadge.textContent = `${count} memories`;
    }
  }

  function renderMemoriesList() {
    if (!elements.memoriesItemsList) return;
    elements.memoriesItemsList.innerHTML = '';

    if (state.memories.length === 0) {
      elements.memoriesItemsList.innerHTML = `
        <div class="empty-files-hint">
          <p>No long-term memories stored yet.</p>
          <span style="font-size: 11px; opacity: 0.7;">The chatbot automatically learns facts and preferences, or you can add them above.</span>
        </div>
      `;
      return;
    }

    state.memories.forEach(mem => {
      const row = document.createElement('div');
      row.className = 'memory-row-card';
      const catClass = `memory-cat-${mem.category || 'general'}`;
      const dateStr = mem.created_at ? new Date(mem.created_at).toLocaleDateString() : '';

      row.innerHTML = `
        <div class="memory-row-left">
          <span class="memory-category-tag ${catClass}">${escapeHtml(mem.category || 'general')}</span>
          <span class="memory-content-text">${escapeHtml(mem.content)}</span>
          <span class="memory-meta-sub">${dateStr ? `Added ${dateStr}` : ''}</span>
        </div>
        <button class="btn-memory-delete" data-id="${escapeHtml(mem.id)}" title="Delete memory">
          <i data-lucide="trash-2"></i>
        </button>
      `;

      const delBtn = row.querySelector('.btn-memory-delete');
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const memId = delBtn.getAttribute('data-id');
        await deleteMemoryItem(memId);
      });

      elements.memoriesItemsList.appendChild(row);
    });

    if (window.lucide) window.lucide.createIcons();
  }

  async function addMemoryItem() {
    const text = elements.newMemoryInput?.value.trim();
    if (!text) return;
    const cat = elements.newMemoryCategory?.value || 'preference';

    try {
      const res = await fetch('/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text, category: cat })
      });
      if (res.ok) {
        elements.newMemoryInput.value = '';
        await fetchMemoriesList();
      } else {
        const err = await res.json();
        alert(`Failed to add memory: ${err.detail}`);
      }
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  }

  async function deleteMemoryItem(memId) {
    try {
      const res = await fetch(`/memory/${encodeURIComponent(memId)}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        await fetchMemoriesList();
      }
    } catch (e) {
      console.error('Failed to delete memory:', e);
    }
  }

  async function clearAllMemories() {
    if (!confirm('Are you sure you want to clear all long-term memories?')) return;
    try {
      const res = await fetch('/memory', { method: 'DELETE' });
      if (res.ok) {
        await fetchMemoriesList();
      }
    } catch (e) {
      console.error('Failed to clear memories:', e);
    }
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Sidebar toggle
    function toggleSidebar() {
      if (window.innerWidth <= 768) {
        const isOpen = elements.sidebar.classList.toggle('mobile-open');
        elements.sidebarBackdrop?.classList.toggle('hidden', !isOpen);
      } else {
        const isCollapsed = elements.sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar_open', !isCollapsed ? 'true' : 'false');
      }
    }

    function closeSidebar() {
      if (elements.sidebar) {
        elements.sidebar.classList.add('collapsed');
        elements.sidebar.classList.remove('mobile-open');
        elements.sidebarBackdrop?.classList.add('hidden');
        localStorage.setItem('sidebar_open', 'false');
      }
    }

    elements.sidebarCollapseBtn?.addEventListener('click', closeSidebar);
    elements.sidebarOpenBtn?.addEventListener('click', toggleSidebar);
    elements.sidebarBackdrop?.addEventListener('click', closeSidebar);

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
    });

    // New Chat
    elements.newChatBtn?.addEventListener('click', () => {
      createNewConversation();
      if (window.innerWidth <= 768) {
        closeSidebar();
      }
    });

    // Plus Upload Button on Input Box
    elements.inputPlusBtn?.addEventListener('click', () => {
      elements.directFileInput?.click();
    });

    elements.directFileInput?.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadFileDirectly(e.target.files[0]);
        e.target.value = '';
      }
    });

    // Files Button in Header
    elements.headerFilesBtn?.addEventListener('click', () => {
      fetchDocumentsList();
      openModal(elements.filesModal);
    });

    elements.closeFilesModalBtn?.addEventListener('click', () => closeModal(elements.filesModal));
    elements.closeFilesBtn?.addEventListener('click', () => closeModal(elements.filesModal));

    // Memory Button in Header
    elements.headerMemoryBtn?.addEventListener('click', () => {
      fetchMemoriesList();
      openModal(elements.memoryModal);
    });

    elements.closeMemoryModalBtn?.addEventListener('click', () => closeModal(elements.memoryModal));
    elements.closeMemoryBtn?.addEventListener('click', () => closeModal(elements.memoryModal));
    elements.addMemoryBtn?.addEventListener('click', addMemoryItem);
    elements.newMemoryInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        addMemoryItem();
      }
    });
    elements.clearAllMemoriesBtn?.addEventListener('click', clearAllMemories);

    // Modal Dropzone
    elements.modalDropzone?.addEventListener('click', () => {
      elements.modalFileInput?.click();
    });

    elements.modalFileInput?.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadFileDirectly(e.target.files[0]);
        e.target.value = '';
      }
    });

    // Chat input auto-expand
    elements.chatInput?.addEventListener('input', () => {
      autoResizeTextarea(elements.chatInput);
      const len = elements.chatInput.value.length;
      elements.sendBtn.disabled = len === 0 || state.isGenerating;
    });

    elements.chatInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    elements.sendBtn?.addEventListener('click', () => sendMessage());

    // Settings Modal
    elements.openSettingsBtn?.addEventListener('click', () => openModal(elements.settingsModal));
    elements.closeSettingsModalBtn?.addEventListener('click', () => closeModal(elements.settingsModal));

    bindSlider(elements.cfgTemperature, elements.cfgTemperatureVal);
    bindSlider(elements.cfgConfidence, elements.cfgConfidenceVal);

    elements.saveSettingsBtn?.addEventListener('click', () => {
      state.settings.model = elements.cfgModelSelect.value;
      state.settings.temperature = parseFloat(elements.cfgTemperature.value);
      state.settings.confidence = parseFloat(elements.cfgConfidence.value);
      localStorage.setItem('clean_rag_settings', JSON.stringify(state.settings));
      closeModal(elements.settingsModal);
    });

    // Dismiss chat options menu on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.conversation-actions')) {
        document.querySelectorAll('.chat-options-menu').forEach(m => m.classList.add('hidden'));
      }
    });
  }

  function bindSlider(slider, display) {
    if (!slider || !display) return;
    slider.addEventListener('input', () => {
      display.textContent = slider.value;
    });
  }

  function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  }

  function openModal(modal) {
    modal.classList.remove('hidden');
    if (window.lucide) window.lucide.createIcons();
  }

  function closeModal(modal) {
    modal.classList.add('hidden');
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  // --- CONVERSATION MANAGEMENT ---
  function createNewConversation() {
    const newId = 'conv_' + Date.now().toString(36);
    const newConv = {
      id: newId,
      title: 'New chat',
      createdAt: new Date().toISOString(),
      messages: []
    };
    state.conversations.unshift(newConv);
    saveConversations();
    selectConversation(newId);
  }

  function selectConversation(id) {
    state.currentConversationId = id;
    const conv = getCurrentConversation();
    if (conv) {
      elements.currentChatTitle.textContent = conv.title;
      renderConversationsList();
      renderMessages();
      if (window.innerWidth <= 768) {
        elements.sidebar.classList.remove('mobile-open');
        elements.sidebarBackdrop?.classList.add('hidden');
      }
    }
  }

  function getCurrentConversation() {
    return state.conversations.find(c => c.id === state.currentConversationId);
  }

  function saveConversations() {
    localStorage.setItem('clean_rag_conversations', JSON.stringify(state.conversations));
    renderConversationsList();
  }

  function loadConversations() {
    const saved = localStorage.getItem('clean_rag_conversations');
    if (saved) {
      try {
        state.conversations = JSON.parse(saved);
      } catch (e) {
        state.conversations = [];
      }
    }
  }

  function loadSavedSettings() {
    const saved = localStorage.getItem('clean_rag_settings');
    if (saved) {
      try {
        state.settings = { ...state.settings, ...JSON.parse(saved) };
      } catch (e) {}
    }
  }

  function deleteConversation(id) {
    const idx = state.conversations.findIndex(c => c.id === id);
    if (idx === -1) return;

    state.conversations.splice(idx, 1);
    localStorage.setItem('clean_rag_conversations', JSON.stringify(state.conversations));

    if (state.currentConversationId === id) {
      if (state.conversations.length > 0) {
        const nextId = state.conversations[Math.min(idx, state.conversations.length - 1)].id;
        selectConversation(nextId);
      } else {
        createNewConversation();
      }
    } else {
      renderConversationsList();
    }
  }

  function renderConversationsList() {
    elements.conversationList.innerHTML = '';
    state.conversations.forEach(conv => {
      const item = document.createElement('div');
      item.className = `conversation-item ${conv.id === state.currentConversationId ? 'active' : ''}`;
      item.setAttribute('data-id', conv.id);
      item.innerHTML = `
        <span class="conversation-title" title="${escapeHtml(conv.title)}">${escapeHtml(conv.title)}</span>
        <div class="conversation-actions">
          <button class="btn-chat-delete-icon" title="Delete chat" type="button" data-id="${conv.id}">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
          </button>
        </div>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.closest('.conversation-actions')) return;
        selectConversation(conv.id);
      });

      const delBtn = item.querySelector('.btn-chat-delete-icon');
      delBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConversation(conv.id);
      });

      elements.conversationList.appendChild(item);
    });

    if (window.lucide) window.lucide.createIcons();
  }

  // --- MESSAGES RENDERING ---
  function renderMessages() {
    const conv = getCurrentConversation();
    if (!conv || conv.messages.length === 0) {
      elements.welcomeHero.classList.remove('hidden');
      elements.messagesList.innerHTML = '';
      return;
    }

    elements.welcomeHero.classList.add('hidden');
    elements.messagesList.innerHTML = '';

    conv.messages.forEach(msg => {
      appendMessageToDOM(msg);
    });

    scrollToBottom();
  }

  function appendMessageToDOM(msg) {
    const row = document.createElement('div');
    row.className = `message-row ${msg.role}`;

    const isUser = msg.role === 'user';
    let formattedContent = msg.content;
    if (!isUser && window.marked) {
      formattedContent = marked.parse(msg.content);
    } else {
      formattedContent = `<p>${escapeHtml(msg.content)}</p>`;
    }

    let sourcesHtml = '';
    if (msg.sources && msg.sources.length > 0) {
      sourcesHtml = `
        <div class="sources-container">
          <div class="sources-header">Sources</div>
          <div class="citation-chips">
            ${msg.sources.map(s => `
              <div class="citation-card">
                <i data-lucide="link" style="width: 11px; height: 11px;"></i>
                <span>${escapeHtml(s.document)}${s.page ? ' (p. ' + s.page + ')' : ''}</span>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    const copyAction = !isUser ? `
      <div class="message-footer-actions">
        <button class="msg-action-btn copy-msg-btn"><i data-lucide="copy" style="width: 12px; height: 12px;"></i> Copy</button>
        ${msg.latency ? `<span>${msg.latency}s</span>` : ''}
      </div>
    ` : '';

    row.innerHTML = `
      <div class="message-body">
        <div class="message-bubble">
          ${formattedContent}
          ${sourcesHtml}
        </div>
        ${copyAction}
      </div>
    `;

    const copyBtn = row.querySelector('.copy-msg-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(msg.content);
        copyBtn.textContent = 'Copied';
        setTimeout(() => {
          copyBtn.innerHTML = '<i data-lucide="copy" style="width: 12px; height: 12px;"></i> Copy';
          if (window.lucide) window.lucide.createIcons();
        }, 1500);
      });
    }

    elements.messagesList.appendChild(row);
    if (window.lucide) window.lucide.createIcons();
  }

  function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
  }

  // --- SEND MESSAGE ---
  async function sendMessage() {
    const text = elements.chatInput.value.trim();
    if (!text || state.isGenerating) return;

    const conv = getCurrentConversation();
    if (!conv) return;

    if (conv.messages.length === 0) {
      conv.title = text.length > 28 ? text.slice(0, 28) + '...' : text;
      elements.currentChatTitle.textContent = conv.title;
      saveConversations();
    }

    const userMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString()
    };

    conv.messages.push(userMessage);
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    elements.sendBtn.disabled = true;
    state.isGenerating = true;

    renderMessages();
    elements.agentThinkingCard.classList.remove('hidden');
    scrollToBottom();

    const startTime = performance.now();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conv.id,
          message: text,
          stream: false
        })
      });

      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      const latency = ((performance.now() - startTime) / 1000).toFixed(2);

      conv.messages.push({
        role: 'assistant',
        content: data.answer || '',
        agent: data.agent || 'general_agent',
        sources: data.sources || [],
        latency: data.latency_seconds || latency,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      conv.messages.push({
        role: 'assistant',
        content: `Error: ${err.message}`,
        timestamp: new Date().toISOString()
      });
    } finally {
      elements.agentThinkingCard.classList.add('hidden');
      state.isGenerating = false;
      elements.sendBtn.disabled = elements.chatInput.value.trim().length === 0;
      saveConversations();
      renderMessages();
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
  }
});
