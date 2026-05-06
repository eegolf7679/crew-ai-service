# crew-ai-service

A minimal FastAPI wrapper around CrewAI, deployed on Railway. The service
exposes a generic `/run` endpoint that the **CIO KB v2** application calls
to dispatch named crews.

This repo intentionally contains **no agents or tools** — the consumer app
wires those in. Here you only get:

- FastAPI app with bearer-token auth
- `/health` for Railway probes
- `/run` skeleton that accepts `{crew, inputs, company, model}`
- Dockerfile + `railway.json` for one-click deploy

## Endpoints

| Method | Path     | Auth   | Description |
|--------|----------|--------|-------------|
| GET    | /health  | none   | Liveness |
| GET    | /        | none   | Service info |
| POST   | /run     | Bearer | Run a named crew |

`POST /run` body:

```json
{
  "crew": "research",
  "inputs": {"question": "..."},
  "company": "Acme Corp",
  "model": "google/gemini-2.5-pro"
}
```

## Environment variables

| Var | Required | Notes |
|---|---|---|
| `SHARED_TOKEN` | yes | Bearer token callers must present |
| `LOVABLE_API_KEY` | later | LLM key (consumer wires this in) |
| `LOVABLE_AI_BASE_URL` | optional | Defaults to Lovable AI Gateway |
| `DEFAULT_MODEL` | optional | Default LLM model |
| `PORT` | optional | Railway sets this automatically |
| `LOG_LEVEL` | optional | `info` by default |

## Deploy

1. Push to GitHub.
2. Railway → New Project → Deploy from repo.
3. Set `SHARED_TOKEN` in Variables.
4. Networking → Generate Domain.
5. `curl https://<domain>/health` → `{"status":"ok"}`.