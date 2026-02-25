/* === Translator UI — App Logic === */

// ─── DOM refs ────────────────────────────────────────────────
const sourceText = document.getElementById('sourceText');
const modelOutput = document.getElementById('modelOutput');
const googleOutput = document.getElementById('googleOutput');
const modelBleu = document.getElementById('modelBleu');
const backTranslationContainer = document.getElementById('backTranslationContainer');
const backTranslationDisplay = document.getElementById('backTranslationDisplay');
const tokenToggleBtn = document.getElementById('tokenToggleBtn');
const tokenDisplay = document.getElementById('tokenDisplay');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const swapBtn = document.getElementById('swapBtn');
const srcLangBtn = document.getElementById('srcLangBtn');
const tgtLangBtn = document.getElementById('tgtLangBtn');
const translateBtn = document.getElementById('translateBtn');
const loadingBar = document.getElementById('loadingBar');

// ─── State ───────────────────────────────────────────────────
let debounceTimer = null;
let isTranslating = false;
let currentTranslation = ''; // Will store model translation
let showTokens = false;
const DEBOUNCE_MS = 800;

// ─── Translate ───────────────────────────────────────────────
async function translate() {
  const text = sourceText.value.trim();

  if (!text) {
    modelOutput.innerHTML = '<span class="placeholder-text">Model translation...</span>';
    googleOutput.innerHTML = '<span class="placeholder-text">Google translation...</span>';
    modelBleu.textContent = 'BLEU: --';
    backTranslationContainer.style.display = 'none';
    tokenDisplay.innerHTML = '';
    tokenDisplay.style.display = 'none';
    currentTranslation = '';
    return;
  }

  if (isTranslating) return;
  isTranslating = true;
  loadingBar.classList.add('active');

  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      modelOutput.textContent = `Error: ${err.error || 'Unknown error'}`;
      googleOutput.textContent = `Error: ${err.error || 'Unknown error'}`;
      return;
    }

    const data = await res.json();

    // Source Tokens
    if (data.src_tokens) {
      tokenDisplay.innerHTML = data.src_tokens
        .map(t => `<span class="token-pill">${t.replace(' ', '·')}</span>`)
        .join('');
    }

    // Model results
    if (data.model) {
      currentTranslation = data.model.translation || '';
      modelOutput.textContent = currentTranslation || 'No translation from model';
      modelBleu.textContent = data.model.bleu !== null ? `BLEU: ${data.model.bleu}` : 'BLEU: --';

      if (data.model.back_translation) {
        backTranslationContainer.style.display = 'block';
        backTranslationDisplay.textContent = data.model.back_translation;
      } else {
        backTranslationContainer.style.display = 'none';
      }

      if (data.model.error) modelOutput.textContent = `Error: ${data.model.error}`;
    }

    // Google results
    if (data.google) {
      googleOutput.textContent = data.google.translation || 'No translation from Google';
      if (data.google.error) googleOutput.textContent = `Error: ${data.google.error}`;
    }

  } catch (e) {
    modelOutput.textContent = 'Could not connect to server.';
    console.error('[Translate]', e);
  } finally {
    isTranslating = false;
    loadingBar.classList.remove('active');
  }
}

// ─── Debounced translate on typing ───────────────────────────
sourceText.addEventListener('input', () => {
  updateCharCount();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(translate, DEBOUNCE_MS);
});

// ─── Char count ──────────────────────────────────────────────
function updateCharCount() {
  const len = sourceText.value.length;
  charCount.textContent = `${len.toLocaleString()} / 5,000`;
}

// ─── Clear ───────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  sourceText.value = '';
  modelOutput.innerHTML = '<span class="placeholder-text">Model translation...</span>';
  googleOutput.innerHTML = '<span class="placeholder-text">Google translation...</span>';
  modelBleu.textContent = 'BLEU: --';
  backTranslationContainer.style.display = 'none';
  tokenDisplay.innerHTML = '';
  tokenDisplay.style.display = 'none';
  tokenToggleBtn.classList.remove('active');
  tokenToggleBtn.textContent = 'Show Tokens';
  showTokens = false;
  currentTranslation = '';
  updateCharCount();
  sourceText.focus();
});

// ─── Token Toggle ────────────────────────────────────────────
tokenToggleBtn.addEventListener('click', () => {
  showTokens = !showTokens;
  tokenToggleBtn.classList.toggle('active', showTokens);
  tokenToggleBtn.textContent = showTokens ? 'Hide Tokens' : 'Show Tokens';
  tokenDisplay.style.display = showTokens ? 'flex' : 'none';
});

// ─── Copy ────────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  if (!currentTranslation) return;
  try {
    await navigator.clipboard.writeText(currentTranslation);
    copyBtn.classList.add('copied');
    setTimeout(() => copyBtn.classList.remove('copied'), 1500);
  } catch (_) { }
});

// ─── Swap languages ──────────────────────────────────────────
swapBtn.addEventListener('click', () => {
  // Swap button labels
  const tmp = srcLangBtn.textContent;
  srcLangBtn.textContent = tgtLangBtn.textContent;
  tgtLangBtn.textContent = tmp;

  // Move translation to source, clear target
  if (currentTranslation) {
    sourceText.value = currentTranslation;
    modelOutput.innerHTML = '<span class="placeholder-text">Model translation...</span>';
    googleOutput.innerHTML = '<span class="placeholder-text">Google translation...</span>';
    modelBleu.textContent = 'BLEU: --';
    tokenDisplay.innerHTML = '';
    tokenDisplay.style.display = 'none';
    tokenToggleBtn.classList.remove('active');
    tokenToggleBtn.textContent = 'Show Tokens';
    showTokens = false;
    currentTranslation = '';
    updateCharCount();
    // Auto-translate the swapped text
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(translate, 300);
  }
});

// ─── Translate button (mobile) ───────────────────────────────
translateBtn.addEventListener('click', translate);

// ─── Keyboard shortcut: Ctrl+Enter to translate ──────────────
sourceText.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    clearTimeout(debounceTimer);
    translate();
  }
});

// ─── Init ────────────────────────────────────────────────────
updateCharCount();
