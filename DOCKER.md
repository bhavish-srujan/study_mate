# Running StudyMate AI with Docker

This project runs as a single Streamlit container. The container includes the
application and its Python dependencies, but **not** your `.env`, your indexed
library, or the BGE-M3 embedding model. Those are kept outside the image.

## What you need

- Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux)
- About 6 GB of free disk space (the image is ~3 GB; the model cache is ~2.3 GB,
  plus Chroma data and uploads)
- 8 GB RAM recommended; BGE-M3 is a large local embedding model
- A Groq API key for chat, quiz generation, and short-answer grading

## Quick start

From the project folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set at least:

```dotenv
LLM_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Then build and start:

```powershell
docker compose up --build
```

Open **http://localhost:8501**. If the local Python app is already using that
port, set `STUDYMATE_PORT=8502` in `.env` (or `$env:STUDYMATE_PORT = "8502"`)
and run `docker compose up -d`; then open `http://localhost:8502`.

The first search or indexing run downloads `BAAI/bge-m3` (~2.3 GB) into the
`studymate_data` volume. Later starts reuse it. The first search after a restart
also loads the model into RAM and can take 20–35 seconds on CPU.

Useful commands:

```powershell
docker compose logs -f          # follow logs
docker compose down             # stop, keep indexed data
docker compose down -v          # stop and delete the data/model volume
docker compose ps               # container health
curl.exe -fsS http://localhost:8501/_stcore/health
```

## Configuration

The compose file passes `.env` into the container at runtime. It is never copied
into the image. The container overrides these paths so everything persists on the
`studymate_data` volume:

| Variable | Container value |
| --- | --- |
| `CHROMA_DIR` | `/app/data/chroma` |
| `SQLITE_PATH` | `/app/data/studymate.db` |
| `HF_HOME` | `/app/data/huggingface` |
| `TRANSFORMERS_CACHE` | `/app/data/huggingface` |
| `SENTENCE_TRANSFORMERS_HOME` | `/app/data/huggingface` |
| `EMBEDDING_DEVICE` | `cpu` |

The container is CPU-only by default. If you have a GPU and want to use it, you
need a CUDA-enabled PyTorch base image and an `EMBEDDING_DEVICE=cuda` override;
the included Dockerfile is intentionally the portable CPU version.

## Sending the project to a friend

### Option 1 — GitHub (best if you may update the code)

1. Push the project to a private or public GitHub repository.
2. Make sure `.env`, `data/`, `.venv/`, and caches are **not** committed. The
   repository's `.gitignore` already excludes them.
3. Your friend clones the repository, copies `.env.example` to `.env`, adds
   their own Groq key, and runs:

```powershell
docker compose up --build
```

This is the cleanest option: they get the code and build the image locally.

### Option 2 — Docker Hub (best for a non-technical friend)

Build and push an image:

```powershell
docker login
docker build -t YOUR_DOCKERHUB_NAME/studymate-ai:latest .
docker push YOUR_DOCKERHUB_NAME/studymate-ai:latest
```

Your friend creates a `.env` file and runs:

```powershell
docker pull YOUR_DOCKERHUB_NAME/studymate-ai:latest
docker run --rm -p 8501:8501 --env-file .env `
  -v studymate_data:/app/data `
  YOUR_DOCKERHUB_NAME/studymate-ai:latest
```

Then open **http://localhost:8501**. The first run still downloads the embedding
model into the named volume. Send your friend the project's `.env.example` too
(it contains no secrets) so they can create `.env` without guessing the variable
names.

### Option 3 — Save the image as a file

This works without a registry, but the image is large (often 2 GB+) and the
embedding model is downloaded separately on first run:

```powershell
docker save -o studymate-ai.tar studymate-ai:latest
```

Send `studymate-ai.tar` and `.env.example` through OneDrive, Google Drive, etc.
Your friend loads the image and creates `.env` from the template:

```powershell
docker load -i studymate-ai.tar
docker run --rm -p 8501:8501 --env-file .env `
  -v studymate_data:/app/data `
  studymate-ai:latest
```

### Option 4 — Host it and share a URL

You can run the container on a VPS and open port `8501`, but the app has **no
login or authentication**. Do not expose it directly to the public internet.
Use a private network such as Tailscale, or put it behind a reverse proxy with
authentication and HTTPS. For a quick temporary share, a Cloudflare Tunnel or
ngrok can work, but anyone with the URL can use your Groq key.

## What not to send

Do not send these to your friend:

- `.env` — contains your Groq API key
- `data/` — your Chroma index, SQLite database, uploads, and model cache
- `.venv/` — your local Python environment
- `__pycache__/`, `.pytest_cache/` — generated caches

The included `.dockerignore` prevents `.env` and `data/` from entering the image.
The `.gitignore` prevents them from entering Git.

If you deliberately want to share your indexed library too, send the `data/`
folder separately and have your friend copy it into the Docker volume. It may be
large, and it contains your study material.

## Troubleshooting

**`env file .env not found`**  
Copy `.env.example` to `.env` first.

**Port 8501 is already in use**  
Set a different host port, then restart:

```powershell
$env:STUDYMATE_PORT = "8502"
docker compose up -d
```

Then open `http://localhost:8502`. You can also put `STUDYMATE_PORT=8502` in
`.env`.

**The first search is very slow**  
The BGE-M3 model is downloading or loading. Watch progress with
`docker compose logs -f`. Later searches reuse the model and embedding cache.

**Out of memory**  
BGE-M3 needs several GB of RAM. Increase Docker Desktop's memory limit or close
other applications. You can lower `EMBEDDING_BATCH_SIZE` in `.env`.

**Apple Silicon (M-series Mac)**  
If the CPU PyTorch wheel is unavailable for your architecture, add this to the
`studymate` service in `docker-compose.yml` and rebuild:

```yaml
platform: linux/amd64
```

This runs the image under emulation and will be slower.

**YouTube says no transcript**  
The video needs an English caption track. The loader requests common English
variants (`en`, `en-US`, `en-GB`, etc.). If none exist, the error lists the
languages the video actually has. YouTube may also block transcript requests
from some cloud/VPS IP addresses.

**I need to see the full error**  
Run `docker compose logs -f` and retry the action in the UI.
