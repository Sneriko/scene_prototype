# RunPod deployment

The repository includes a production-style container for a **RunPod Pod**. A Pod
is used rather than a Serverless endpoint because this project exposes a
long-running FastAPI service and browser UI.

## 1. Build and publish the image

From the repository root, choose a registry name and build the image:

```bash
docker build -t <registry-user>/ambulance-case:latest .
docker push <registry-user>/ambulance-case:latest
```

Do not bake API tokens into the image or pass them as Docker build arguments.
Configure them as encrypted environment variables in RunPod instead.

## 2. Create the Pod

In the RunPod console, create a Pod from the published image and expose **HTTP
port 8080**. The container binds to `0.0.0.0`, and RunPod's HTTP proxy can route
to that port. The root URL serves the demo UI and `/health` is the health-check
endpoint.

The checked-in configuration defaults to fully local processing:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | HTTP port inside the container. |
| `TRANSCRIPTION_BACKEND` | `local_edge` | `local_edge`, `local_kb_whisper`, or `openai`. |
| `KB_WHISPER_SIZE` | `large` | KB Whisper model size (`tiny`, `base`, `small`, `medium`, or `large`). |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | OpenAI-compatible local inference endpoint. |
| `LOCAL_LLM_MODEL` | `qwen2.5-7b-instruct` | Model exposed by the local endpoint. |
| `LOCAL_LLM_API_KEY` | `local` | Local endpoint credential, if required. |
| `HUGGINGFACE_TOKEN` | unset | Token for gated pyannote weights. |
| `OPENAI_API_KEY` | unset | Required only for an OpenAI-backed mode. |

For `local_edge`, the Pod must also run or be able to reach an
OpenAI-compatible LLM server. Set `LOCAL_LLM_BASE_URL` to that service. Note
that `127.0.0.1` refers to this application container; use an address reachable
from the container if the model server runs elsewhere.

For a UI-only demonstration using the prepared outputs, the app can start
without model credentials. Live recording processing requires the selected ASR
and generation backends to be configured.

## 3. Storage and model cache

Attach a persistent RunPod network volume when model downloads or live case
artifacts must survive a Pod restart. The app writes uploaded live cases below
`/app/edge_cases`. Model caches use the normal Hugging Face/PyTorch cache
locations; point `HF_HOME` and `TORCH_HOME` at directories on the mounted volume
to avoid downloading weights on every new Pod.

Example settings for a volume mounted at `/runpod-volume`:

```text
HF_HOME=/runpod-volume/huggingface
TORCH_HOME=/runpod-volume/torch
```

The prepared demo data under `/app/data` and `/app/outputs` is copied into the
image and is available without a volume.

## 4. Verify the deployment

After the Pod is healthy, use its RunPod proxy URL:

```bash
curl -f https://<pod-proxy-host>/health
```

A successful response reports `"status":"ok"`. Open the proxy root URL in a
browser to use the demo UI.

## Security note

The sample recordings and outputs are included in the container image. Verify
that they contain no sensitive data before publishing the image. Never publish
real patient audio or credentials in an image; use a private registry, encrypted
RunPod secrets, and an appropriate retention policy for any clinical workload.
