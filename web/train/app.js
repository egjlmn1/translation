/* === Training Dashboard — Live WebSocket + Chart.js === */

// ─── State ───────────────────────────────────────────────────
let ws = null;
let lossChart, bleuChart, lrChart, speedChart;
let reconnectTimer = null;

// ─── DOM refs ────────────────────────────────────────────────
const statusBadge = document.getElementById('statusBadge');
const epochValue = document.getElementById('epochValue');
const stepValue = document.getElementById('stepValue');
const speedValue = document.getElementById('speedValue');
const etaValue = document.getElementById('etaValue');
const lossValue = document.getElementById('lossValue');
const bleuValue = document.getElementById('bleuValue');
const samplesContainer = document.getElementById('samplesContainer');

// ─── Chart setup ─────────────────────────────────────────────
Chart.defaults.color = '#8888a0';
Chart.defaults.borderColor = '#2a2a3a';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

function createChart(canvasId, label, borderColor, bgColor) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: label,
        data: [],
        borderColor: borderColor,
        backgroundColor: bgColor,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Step', color: '#666' },
          ticks: { maxTicksLimit: 10 },
          grid: { color: 'rgba(255,255,255,0.03)' },
        },
        y: {
          display: true,
          grid: { color: 'rgba(255,255,255,0.03)' },
        }
      },
      plugins: {
        legend: { display: false },
      }
    }
  });
}

function initCharts() {
  // Loss chart has two datasets (train + val)
  const lossCtx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Train Loss',
          data: [],
          borderColor: '#6c5ce7',
          backgroundColor: 'rgba(108,92,231,0.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
        },
        {
          label: 'Val Loss',
          data: [],
          borderColor: '#ff9100',
          backgroundColor: 'rgba(255,145,0,0.06)',
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.3,
          fill: true,
          borderDash: [6, 3],
          spanGaps: true,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      spanGaps: true, // Also set global for train loss
      animation: { duration: 300 },
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          title: { display: true, text: 'Step', color: '#666' },
          ticks: { maxTicksLimit: 12 },
          grid: { color: 'rgba(255,255,255,0.03)' },
        },
        y: {
          title: { display: true, text: 'Loss', color: '#666' },
          grid: { color: 'rgba(255,255,255,0.03)' },
        }
      },
      plugins: {
        legend: {
          display: true,
          labels: { boxWidth: 12, padding: 16 },
        },
      }
    }
  });

  bleuChart = createChart('bleuChart', 'BLEU', '#448aff', 'rgba(68,138,255,0.08)');
  lrChart = createChart('lrChart', 'Learning Rate', '#00e676', 'rgba(0,230,118,0.08)');
  speedChart = createChart('speedChart', 'Samples/sec', '#ff3d00', 'rgba(255,61,0,0.08)');
}

// ─── Update UI ───────────────────────────────────────────────
function updateDashboard(data) {
  // Stats
  const currentEpoch = data.epoch + 1;
  const totalEpochs = data.total_epochs || '—';
  epochValue.textContent = `${currentEpoch} / ${totalEpochs}`;
  stepValue.textContent = data.global_step.toLocaleString();
  speedValue.textContent = data.samples_per_sec > 0 ? `${data.samples_per_sec} samp/s` : '—';
  etaValue.textContent = formatETA(data.eta_seconds);

  // Status badge
  if (data.training_active) {
    statusBadge.textContent = 'Training';
    statusBadge.className = 'status-badge training';
  } else if (data.global_step > 0) {
    statusBadge.textContent = 'Stopped';
    statusBadge.className = 'status-badge connected';
  }

  // Latest values
  if (data.train_losses.length > 0) {
    const last = data.train_losses[data.train_losses.length - 1];
    lossValue.textContent = last.loss.toFixed(4);
  }
  if (data.bleu_scores.length > 0) {
    const last = data.bleu_scores[data.bleu_scores.length - 1];
    bleuValue.textContent = last.bleu.toFixed(4);
  }

  // Loss chart
  const trainLabels = data.train_losses.map(d => d.step);
  const trainData = data.train_losses.map(d => d.loss);
  lossChart.data.labels = trainLabels;
  lossChart.data.datasets[0].data = trainData;

  // Val loss — align to train x-axis
  const valMap = {};
  data.val_losses.forEach(d => { valMap[d.step] = d.loss; });
  lossChart.data.datasets[1].data = trainLabels.map(s => valMap[s] ?? null);
  lossChart.update('none');

  // BLEU chart
  bleuChart.data.labels = data.bleu_scores.map(d => d.step);
  bleuChart.data.datasets[0].data = data.bleu_scores.map(d => d.bleu);
  bleuChart.update('none');

  // LR chart
  lrChart.data.labels = data.learning_rates.map(d => d.step);
  lrChart.data.datasets[0].data = data.learning_rates.map(d => d.lr);
  lrChart.update('none');

  // Speed chart
  if (data.speed_history) {
    speedChart.data.labels = data.speed_history.map(d => d.step);
    speedChart.data.datasets[0].data = data.speed_history.map(d => d.speed);
    speedChart.update('none');
  }

  // Sample translations
  if (data.sample_translations.length > 0) {
    samplesContainer.innerHTML = '';
    // Show most recent first
    const samples = [...data.sample_translations].reverse();
    for (const s of samples) {
      const el = document.createElement('div');
      el.className = 'sample-item';
      el.innerHTML = `
        <div class="label src">Source (step ${s.step})</div>
        <div class="text">${escapeHtml(s.src)}</div>
        <div class="label ref">Reference</div>
        <div class="text">${escapeHtml(s.ref)}</div>
        <div class="label pred">Prediction</div>
        <div class="text">${escapeHtml(s.pred)}</div>
      `;
      samplesContainer.appendChild(el);
    }
  }
}

// ─── Helpers ─────────────────────────────────────────────────
function formatETA(seconds) {
  if (!seconds || seconds <= 0) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ─── WebSocket ───────────────────────────────────────────────
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/training`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected');
    statusBadge.textContent = 'Connected';
    statusBadge.className = 'status-badge connected';
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) {
      console.error('[WS] Parse error', e);
    }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected — will reconnect in 3s');
    statusBadge.textContent = 'Disconnected';
    statusBadge.className = 'status-badge';
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

// Also poll status API as fallback (every 5s)
async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      updateDashboard(data);
    }
  } catch (_) { }
}

// ─── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  connect();
  // Also poll as fallback
  setInterval(pollStatus, 5000);
  pollStatus();
});
