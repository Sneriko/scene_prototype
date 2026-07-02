let mediaRecorder;
let recordedChunks = [];
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

async function uploadChunk(blob, extension = 'wav') {
  const formData = new FormData();
  formData.append('file', blob, `chunk-${chunkNumber}.${extension}`);
  const response = await fetch(`/cases/${caseId}/audio-chunks?chunk_number=${chunkNumber}`, {
    method: 'POST',
    body: formData,
  });
  chunkNumber += 1;
  if (!response.ok) throw new Error(await response.text());
}

function encodeWav(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const sampleCount = audioBuffer.length;
  const bytesPerSample = 2;
  const blockAlign = channelCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + sampleCount * blockAlign);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + sampleCount * blockAlign, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, sampleCount * blockAlign, true);

  let offset = 44;
  for (let i = 0; i < sampleCount; i += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += bytesPerSample;
    }
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

async function recordedChunksToWav() {
  const recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(await recordedBlob.arrayBuffer());
    return encodeWav(audioBuffer);
  } finally {
    await audioContext.close();
  }
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
  recordedChunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.start();
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus(`Recording case ${caseId}.`);
});

stopButton.addEventListener('click', async () => {
  mediaRecorder.onstop = async () => {
    try {
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus('Converting recording to WAV...');
      const wavBlob = await recordedChunksToWav();
      setStatus('Uploading recording...');
      await uploadChunk(wavBlob, 'wav');
      setStatus('Finishing recording...');
      const response = await fetch(`/cases/${caseId}/finish-recording`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      pollOutput();
    } catch (error) {
      setStatus(error.message || 'Failed to finish recording.');
    }
  };
  mediaRecorder.stop();
});
