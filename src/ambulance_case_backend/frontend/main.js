let mediaRecorder;
let caseId;
let chunkNumber = 0;
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusBox = document.getElementById('status');
const outputBox = document.getElementById('output');

function setStatus(message) { statusBox.textContent = message; }

async function createCase() {
  const response = await fetch('/cases', { method: 'POST' });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function uploadChunk(blob) {
  const formData = new FormData();
  formData.append('file', blob, `chunk-${chunkNumber}.webm`);
  const response = await fetch(`/cases/${caseId}/audio-chunks?chunk_number=${chunkNumber}`, {
    method: 'POST',
    body: formData,
  });
  chunkNumber += 1;
  if (!response.ok) throw new Error(await response.text());
}

async function pollOutput() {
  const statusResponse = await fetch(`/cases/${caseId}/status`);
  const statusPayload = await statusResponse.json();
  setStatus(`Case ${caseId}: ${statusPayload.status}`);
  if (statusPayload.status === 'ready') {
    const outputResponse = await fetch(`/cases/${caseId}/output`);
    outputBox.textContent = JSON.stringify(await outputResponse.json(), null, 2);
    return;
  }
  if (statusPayload.status === 'failed') {
    outputBox.textContent = statusPayload.error || 'Processing failed.';
    return;
  }
  setTimeout(pollOutput, 1500);
}

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
    const response = await fetch(`/cases/${caseId}/finish-recording`, { method: 'POST' });
    if (!response.ok) throw new Error(await response.text());
    pollOutput();
  }, 500);
});
