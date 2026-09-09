/**
 * ChronoGuard Frontend Application Logic
 * Handles drag-and-drop uploads, asynchronous status polling,
 * Chart.js visualizations, and animated metric counters.
 */

// Helper to format risk colors consistently across app
function getRiskColor(prob) {
  if (prob < 0.33) return '#10B981'; // Emerald
  if (prob < 0.67) return '#F59E0B'; // Amber
  return '#EF4444';                  // Red
}

function getRiskBadgeClass(prob) {
  if (prob < 0.33) return 'badge-safe';
  if (prob < 0.67) return 'badge-warning';
  if (prob < 0.85) return 'badge-danger';
  return 'badge-critical';
}

// Animate numerical counters
function animateValue(element, start, end, duration, decimals = 0, suffix = '') {
  if (!element) return;
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const val = (progress * (end - start) + start).toFixed(decimals);
    element.innerHTML = val + suffix;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

// Upload Page Initialization
function initUploadPage() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const uploadForm = document.getElementById('upload-form');
  const modal = document.getElementById('processing-modal');
  const statusText = document.getElementById('processing-status-text');

  if (!dropzone || !fileInput) return;

  // Drag-and-drop event handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      submitForm(files[0].name);
    }
  });

  dropzone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      submitForm(fileInput.files[0].name);
    }
  });

  function submitForm(filename) {
    if (modal) {
      modal.classList.remove('hidden');
      if (statusText) statusText.innerText = `Ingesting ${filename}...`;
    }

    const formData = new FormData(uploadForm);
    fetch('/upload', {
      method: 'POST',
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        if (data.job_id) {
          pollJobStatus(data.job_id);
        } else if (data.error) {
          alert(`Error: ${data.error}`);
          if (modal) modal.classList.add('hidden');
        }
      })
      .catch(err => {
        alert(`Upload error: ${err}`);
        if (modal) modal.classList.add('hidden');
      });
  }

  // Sample quick buttons
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const sampleName = btn.getAttribute('data-sample');
      if (modal) {
        modal.classList.remove('hidden');
        if (statusText) statusText.innerText = `Running analysis on ${sampleName}...`;
      }

      fetch(`/upload?sample=${encodeURIComponent(sampleName)}`, {
        method: 'POST',
      })
        .then(res => res.json())
        .then(data => {
          if (data.job_id) {
            pollJobStatus(data.job_id);
          } else {
            alert(data.error || 'Failed to trigger sample run.');
            if (modal) modal.classList.add('hidden');
          }
        })
        .catch(err => {
          alert(`Sample load failed: ${err}`);
          if (modal) modal.classList.add('hidden');
        });
    });
  });

  function pollJobStatus(jobId) {
    const interval = setInterval(() => {
      fetch(`/status/${jobId}`)
        .then(res => res.json())
        .then(statusData => {
          if (statusText) {
            if (statusData.status === 'running') {
              statusText.innerText = 'Extracting temporal flow sequences & running LSTM forward pass...';
            } else if (statusData.status === 'done') {
              statusText.innerText = 'Analysis complete! Rendering SOC dashboard...';
            }
          }

          if (statusData.status === 'done') {
            clearInterval(interval);
            setTimeout(() => {
              window.location.href = `/results/${jobId}`;
            }, 600);
          } else if (statusData.status === 'error') {
            clearInterval(interval);
            alert(`Analysis failed: ${statusData.error || 'Unknown error'}`);
            if (modal) modal.classList.add('hidden');
          }
        })
        .catch(err => {
          console.error('Polling error:', err);
        });
    }, 1200);
  }
}

// Results Dashboard Initialization
function initResultsPage(resultsData) {
  if (!resultsData || !resultsData.windows) return;

  const summary = resultsData.summary || {};
  const windows = resultsData.windows || [];

  // 1. Animate metric counters
  const maxProbElem = document.getElementById('metric-max-prob');
  if (maxProbElem) {
    animateValue(maxProbElem, 0, (summary.max_infiltration_probability || 0) * 100, 800, 1, '%');
  }

  const flaggedElem = document.getElementById('metric-flagged-count');
  if (flaggedElem) {
    animateValue(flaggedElem, 0, summary.num_windows_flagged || 0, 800, 0);
  }

  // 2. Timeline Chart.js Line Chart
  const timelineCanvas = document.getElementById('timeline-chart');
  if (timelineCanvas && window.Chart) {
    const labels = windows.map((w, idx) => `W${idx + 1}`);
    const lstmProbs = windows.map(w => w.infiltration_probability);
    const baselineProbs = windows.map(w => w.baseline_probability || 0);

    new Chart(timelineCanvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'ChronoGuard LSTM Forecast',
            data: lstmProbs,
            borderColor: '#22D3EE',
            backgroundColor: 'rgba(34, 211, 238, 0.1)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: lstmProbs.map(p => getRiskColor(p)),
            pointBorderColor: '#0B1120',
            pointRadius: 4,
            pointHoverRadius: 6,
          },
          {
            label: 'Logistic Regression Baseline',
            data: baselineProbs,
            borderColor: '#64748B',
            borderWidth: 1.5,
            borderDash: [5, 5],
            fill: false,
            tension: 0.2,
            pointRadius: 0,
          },
          {
            label: 'Alert Threshold (0.66)',
            data: labels.map(() => 0.66),
            borderColor: 'rgba(239, 68, 68, 0.6)',
            borderWidth: 1,
            borderDash: [2, 4],
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            labels: {
              color: '#94A3B8',
              font: { family: 'Inter', size: 12 }
            }
          },
          tooltip: {
            backgroundColor: '#131B2E',
            titleColor: '#22D3EE',
            bodyColor: '#E5E7EB',
            borderColor: '#1E293B',
            borderWidth: 1,
            callbacks: {
              afterLabel: function(context) {
                const w = windows[context.dataIndex];
                if (w && context.datasetIndex === 0) {
                  return `Predicted Stage: ${w.predicted_stage}`;
                }
                return null;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: '#1E293B' },
            ticks: { color: '#94A3B8', font: { family: 'JetBrains Mono' } }
          },
          y: {
            min: 0.0,
            max: 1.0,
            grid: { color: '#1E293B' },
            ticks: {
              color: '#94A3B8',
              font: { family: 'JetBrains Mono' },
              callback: (val) => `${(val * 100).toFixed(0)}%`
            }
          }
        }
      }
    });
  }

  // 3. Comparison Chart (Baseline vs. LSTM)
  const compCanvas = document.getElementById('comparison-chart');
  if (compCanvas && window.Chart && resultsData.comparison_metrics) {
    const comp = resultsData.comparison_metrics;
    const b = comp.baseline || {};
    const l = comp.lstm || {};

    const metricsLabels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'False Positive Rate'];
    const lstmData = [l.accuracy || 0, l.f1_score || 0, l.precision || 0, l.recall || 0, l.false_positive_rate || 0];
    const baseData = [b.accuracy || 0, b.f1_score || 0, b.precision || 0, b.recall || 0, b.false_positive_rate || 0];

    new Chart(compCanvas, {
      type: 'bar',
      data: {
        labels: metricsLabels,
        datasets: [
          {
            label: 'ChronoGuard LSTM',
            data: lstmData,
            backgroundColor: 'rgba(34, 211, 238, 0.75)',
            borderColor: '#22D3EE',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Logistic Regression Baseline',
            data: baseData,
            backgroundColor: 'rgba(100, 116, 139, 0.65)',
            borderColor: '#64748B',
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: '#94A3B8',
              font: { family: 'Inter', size: 12 }
            }
          },
          tooltip: {
            backgroundColor: '#131B2E',
            borderColor: '#1E293B',
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                return `${context.dataset.label}: ${(context.raw * 100).toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            min: 0,
            max: 1.0,
            grid: { color: '#1E293B' },
            ticks: {
              color: '#94A3B8',
              font: { family: 'JetBrains Mono' },
              callback: (val) => `${(val * 100).toFixed(0)}%`
            }
          }
        }
      }
    });
  }
}

// Auto-run page initializers on DOM load
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('dropzone')) {
    initUploadPage();
  }
});
