let mediaRecorder;
let caseId;
let chunkNumber = 0;
let activeMode = 'demo';
let demoCases = [];
let outputCatalogs = [];

const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusBox = document.getElementById('status');
const runtimeBox = document.getElementById('runtime');
const demoSelect = document.getElementById('demoCases');
const outputCatalogSelect = document.getElementById('outputCatalogs');
const demoAudio = document.getElementById('demoAudio');
const loadDemoButton = document.getElementById('loadDemo');
const demoTab = document.getElementById('demoTab');
const recordTab = document.getElementById('recordTab');
const demoPanel = document.getElementById('demoPanel');
const recordPanel = document.getElementById('recordPanel');
const caseMeta = document.getElementById('caseMeta');
const suggestionsBox = document.getElementById('suggestions');
const journalBox = document.getElementById('journal');
const pdfSection = document.getElementById('pdfSection');
const pdfLinks = document.getElementById('pdfLinks');
const treatmentFrame = document.getElementById('treatmentFrame');
const journalFrame = document.getElementById('journalFrame');

function setStatus(message) { statusBox.textContent = message; }

function setMode(mode) {
  activeMode = mode;
  demoPanel.classList.toggle('hidden', mode !== 'demo');
  recordPanel.classList.toggle('hidden', mode !== 'record');
  demoTab.classList.toggle('active', mode === 'demo');
  recordTab.classList.toggle('active', mode === 'record');
}

async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function loadRuntime() {
  try {
    const health = await jsonFetch('/health');
    runtimeBox.textContent = `${health.backend} backend · local model ${health.local_llm_model}`;
  } catch (error) {
    runtimeBox.textContent = `Backend unavailable: ${error.message}`;
  }
}

async function loadOutputCatalogs() {
  const payload = await jsonFetch('/demo-output-catalogs');
  outputCatalogs = payload.catalogs || [];
  outputCatalogSelect.innerHTML = '';
  outputCatalogs.forEach(catalog => {
    const option = document.createElement('option');
    option.value = catalog.id;
    option.textContent = catalog.label;
    outputCatalogSelect.appendChild(option);
  });
}

function selectedCase() {
  return demoCases.find(item => String(item.case_id) === demoSelect.value);
}

function refreshCatalogOptionsForCase() {
  const currentCase = selectedCase();
  if (!currentCase) return;

  outputCatalogSelect.innerHTML = '';
  outputCatalogs.forEach(catalog => {
    const hasOutput = Boolean(currentCase.outputs && currentCase.outputs[catalog.id]);
    const option = document.createElement('option');
    option.value = catalog.id;
    option.disabled = !hasOutput;
    option.textContent = `${catalog.label} · ${hasOutput ? 'ready' : 'missing output'}`;
    outputCatalogSelect.appendChild(option);
  });

  const firstReady = Array.from(outputCatalogSelect.options).find(option => !option.disabled);
  if (firstReady) outputCatalogSelect.value = firstReady.value;
  loadDemoButton.disabled = !firstReady;
}

async function loadDemoCases() {
  const payload = await jsonFetch('/demo-cases');
  demoCases = payload.cases || [];
  demoSelect.innerHTML = '';
  demoCases.forEach(item => {
    const option = document.createElement('option');
    option.value = item.case_id;
    option.disabled = !item.has_output;
    option.textContent = `${item.label} · ${item.audio_file} · ${item.has_output ? 'ready' : 'missing generated output'}`;
    demoSelect.appendChild(option);
  });
  refreshDemoAudio();
  refreshCatalogOptionsForCase();
}

function refreshDemoAudio() {
  const currentCase = selectedCase();
  if (!currentCase) {
    demoAudio.removeAttribute('src');
    return;
  }
  demoAudio.src = currentCase.audio_url || `/demo-cases/${currentCase.case_id}/audio`;
}

function renderOutput(output, pdfBaseUrl, pdfQuery = '') {
  caseMeta.textContent = `Case ${output.case_id} · ${output.audio_path}`;
  suggestionsBox.innerHTML = '<h3>Suggested treatment instructions</h3>';
  output.treatment_suggestions.forEach(suggestion => {
    const card = document.createElement('article');
    card.className = 'suggestion';
    card.innerHTML = `<span class="badge">${suggestion.urgency || 'review'}</span><h4>${suggestion.title}</h4><p>${suggestion.rationale}</p>`;
    suggestionsBox.appendChild(card);
  });
  journalBox.textContent = output.drafted_journal || 'No journal text returned.';

  const treatmentUrl = `${pdfBaseUrl}/treatment.pdf${pdfQuery}`;
  const journalUrl = `${pdfBaseUrl}/journal.pdf${pdfQuery}`;
  pdfLinks.innerHTML = `
    <a class="button primary" href="${treatmentUrl}" target="_blank" rel="noreferrer">Open treatment PDF</a>
    <a class="button secondary" href="${journalUrl}" target="_blank" rel="noreferrer">Open journal PDF</a>
  `;
  treatmentFrame.src = treatmentUrl;
  journalFrame.src = journalUrl;
  pdfSection.classList.remove('hidden');
}

async function loadSelectedDemo() {
  const selectedCaseId = demoSelect.value;
  const selectedCatalog = outputCatalogSelect.value || 'default';
  if (!selectedCaseId || !selectedCatalog) return;
  const catalog = outputCatalogs.find(item => item.id === selectedCatalog);
  const catalogLabel = catalog ? catalog.label : selectedCatalog;
  setStatus(`Loading demo case ${selectedCaseId} from ${catalogLabel}...`);
  const query = new URLSearchParams({ catalog: selectedCatalog });
  const output = await jsonFetch(`/demo-cases/${selectedCaseId}/output?${query}`);
  renderOutput(output, `/demo-cases/${selectedCaseId}`, `?${query}`);
  setStatus(`Demo case ${selectedCaseId} ready from ${catalogLabel}.`);
}

async function createCase() {
  return jsonFetch('/cases', { method: 'POST' });
}

async function uploadChunk(blob) {
  const formData = new FormData();
  formData.append('file', blob, `chunk-${chunkNumber}.webm`);
  await jsonFetch(`/cases/${caseId}/audio-chunks?chunk_number=${chunkNumber}`, { method: 'POST', body: formData });
  chunkNumber += 1;
}

async function pollOutput() {
  const statusPayload = await jsonFetch(`/cases/${caseId}/status`);
  setStatus(`Case ${caseId}: ${statusPayload.status}`);
  if (statusPayload.status === 'ready') {
    const output = await jsonFetch(`/cases/${caseId}/output`);
    renderOutput(output, `/cases/${caseId}`);
    return;
  }
  if (statusPayload.status === 'failed') {
    journalBox.textContent = statusPayload.error || 'Processing failed.';
    return;
  }
  setTimeout(pollOutput, 1500);
}

demoTab.addEventListener('click', () => setMode('demo'));
recordTab.addEventListener('click', () => setMode('record'));
loadDemoButton.addEventListener('click', () => loadSelectedDemo().catch(error => setStatus(error.message)));
demoSelect.addEventListener('change', () => {
  refreshDemoAudio();
  refreshCatalogOptionsForCase();
});
outputCatalogSelect.addEventListener('change', () => loadSelectedDemo().catch(error => setStatus(error.message)));

startButton.addEventListener('click', async () => {
  const created = await createCase();
  caseId = created.case_id;
  chunkNumber = 0;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  mediaRecorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) uploadChunk(event.data).catch(error => setStatus(error.message));
  };
  mediaRecorder.start(5000);
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus(`Recording case ${caseId}.`);
});

stopButton.addEventListener('click', async () => {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(track => track.stop());
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus('Finishing recording...');
  setTimeout(async () => {
    await jsonFetch(`/cases/${caseId}/finish-recording`, { method: 'POST' });
    pollOutput();
  }, 500);
});

loadRuntime();
Promise.all([loadOutputCatalogs(), loadDemoCases()])
  .then(() => {
    refreshCatalogOptionsForCase();
    return loadSelectedDemo();
  })
  .catch(error => setStatus(error.message));
