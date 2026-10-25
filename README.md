# StudyMate AI

A beginner-friendly RAG application that turns your own study material into
grounded chat answers and practice questions. Built with Streamlit, PyMuPDF,
local sentence-transformer embeddings, ChromaDB, and an LLM.

> **Status: Phase 6 complete; YouTube transcripts and grounded chat ready.**
> PDF, pasted-note, and YouTube transcript ingestion, chunking, local
> embeddings, persistent ChromaDB storage, the resource library UI,
> resource-filtered retrieval, grounded chat, validated question generation,
> interactive quiz taking, scoring, and post-submit answer review are
> implemented. Index your material, ask the chatbot questions, or take a quiz
> and review your score with correct answers, explanations, and source
> references. See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

---

## What works today

- Upload one or more text-based PDFs (up to `MAX_PDFS`, default 5).
- Paste study notes with a custom title.
- Index a YouTube lecture transcript from a watch, `youtu.be`, Shorts, embed, or
  live-video URL. Transcript snippets are grouped into timestamped pages, and
  the timestamp survives chunking for citations.
- Normalize extracted text and split it into overlapping, sentence/paragraph-aware
  chunks that avoid cutting sentences at fixed token boundaries.
- Attach provenance metadata: resource ID, source name/type, page number, and
  YouTube timestamp.
- Embed chunks locally with `BAAI/bge-m3` (1024-dim) and store them in ChromaDB.
- View, count, and delete indexed resources; indexing is idempotent by content.
- Search only selected resources and preview ranked passages in the Generate tab.
- Ask grounded questions in the **Chat** tab. Answers use the same retrieval and
  context-builder stack as quiz generation, and each answer shows its sources,
  including page numbers or YouTube timestamps.
- Choose **Chat with StudyMate first** or **Take the quiz now** before starting a
  generated quiz.
- Deduplicate retrieved chunks and trim the formatted context to a token budget.
- Configure MCQ, fill-in-the-blank, and short-answer questions with a total count and an
  exact easy/medium/hard split (10 questions, 3/5/2 by default).
- Generate grounded questions from a topic (similarity retrieval) or from representative
  chunks across the selected resources (no topic).
- Validate JSON structure, question shapes, counts, source references, selected types, and
  near-duplicates; repair invalid output with up to two feedback calls.
- Take the generated quiz one question at a time with type/difficulty badges, previous/next
  navigation, jump-to-question, and answers that save automatically.
- Confirm submission when questions are unanswered. The answer key, explanations, and
  source references are never exposed before submission.
- Score MCQ and fill-in-the-blank answers by exact match. Normalize fill-in answers
  (case, whitespace, and edge punctuation) and accept configured alternatives.
- Grade short answers approximately with the LLM against the reference answer and key
  points, mapping 0–2 to 0.0/0.5/1.0 points. If the LLM is unavailable, short answers
  are marked **not graded** and excluded from the score instead of penalizing you.
- See total, correct, partial, incorrect, unanswered, score, and percentage, then review
  each question with your answer, the correct answer, an explanation, and source
  references.
- Test the configured Groq LLM from the Generate tab.
- Dark-only interface (no light/dark toggle).
- Friendly errors for empty/scanned PDFs, invalid files, unavailable YouTube
  transcripts, insufficient context, and missing dependencies.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Edit .env: add LLM_API_KEY and set the Groq model (see below)

streamlit run app.py
```

The first indexing run downloads the `BAAI/bge-m3` model (~2.3 GB) to your local
Hugging Face cache. The first search in a new server process imports PyTorch,
loads the model, and can take roughly 20–35 seconds on CPU; later searches reuse
it, and repeated queries are served from an in-process embedding cache. Use
**Warm up search model** in the sidebar after a restart to load it before
searching. Chroma data lives in `data/chroma/` and is gitignored.

BGE-M3 uses a different vector space from the previous `all-MiniLM-L6-v2` index,
so the default collection is now `studymate_bge_m3`. Re-upload your PDFs, notes,
and YouTube transcripts once after switching; the old `studymate_chunks`
collection is left untouched.

### Configuration

All settings are environment variables; see [`.env.example`](.env.example).
The LLM is configured for Groq:

```dotenv
LLM_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0.2
LLM_MAX_COMPLETION_TOKENS=2048
LLM_REASONING_EFFORT=medium
```

The Groq SDK appends `/openai/v1` itself, so `LLM_BASE_URL` accepts either the
host (`https://api.groq.com`) or the full path above; the client normalizes it.
`.env` is gitignored — never commit the key. Use the **Test LLM connection**
button in the Generate tab to verify the key, model, and network.

**Chunking note:** the default is `CHUNK_SIZE_TOKENS=512` with
`CHUNK_OVERLAP_TOKENS=96`. The chunker packs complete sentences and paragraphs
up to that budget, then carries whole trailing sentences into the next chunk.
`EMBEDDING_MAX_SEQ_LENGTH=1024` leaves headroom for BGE-M3's tokenizer while
keeping memory use reasonable. If you raise `CHUNK_SIZE_TOKENS`, raise
`EMBEDDING_MAX_SEQ_LENGTH` at least as high; settings validation enforces this.
`EMBEDDING_BATCH_SIZE=8` is conservative for CPU; increase it on a GPU or a
large-RAM machine. Set `EMBEDDING_DEVICE=cuda` to use a GPU if available.

## Docker

The project includes a CPU-only Docker image and a Compose setup:

```powershell
Copy-Item .env.example .env
notepad .env                 # add LLM_API_KEY and LLM_MODEL
docker compose up --build
```

Open **http://localhost:8501**. If the local Python app is already using that
port, run `$env:STUDYMATE_PORT = "8502"; docker compose up -d` and open
`http://localhost:8502` instead. The `.env` file is passed at runtime and is never
baked into the image. Chroma, SQLite, uploads, and the BGE-M3 model cache persist
in the `studymate_data` volume; the first indexing/search downloads the model
(~2.3 GB). See [`DOCKER.md`](DOCKER.md) for configuration, sharing the app with a
friend, Docker Hub/`docker save` instructions, and troubleshooting.

## UI layout

The app uses one consistent page shell across the four main sections, with a
grouped sidebar:

- **Sidebar:** system status, library counts, a three-step quick start, and an
  **Advanced** area for model warm-up and indexing settings.
- **Library:** ingestion is grouped into **PDFs**, **Pasted notes**, and
  **YouTube** tabs, with indexed resources listed below.
- **Generate:** a compact summary strip, a sectioned quiz-settings form, and a
  side panel for the current quiz or generation notes.
- **Chat:** the transcript sits beside a **Context** panel for resource scope,
  clear-chat, and return-to-quiz controls.
- **Quiz:** the pre-quiz choice, active quiz, and results screens share the same
  headers, quiz summary, and card styling.

## Tests

```powershell
python -m pytest
```

Unit tests use fake embedders, fake retrievers, and mocked LLM responses; no
network calls or API keys are required. The suite currently has **159 tests**
covering chunking, normalization, PDF/pasted-text/YouTube ingestion, the
indexer, ChromaDB storage and representative sampling, a full
ingest→index→delete pipeline, resource-filtered retrieval, context building,
grounded chat, question schemas and validation, generation batching/repair
behavior, the Groq client, quiz answer-vault/view behavior, exact-answer
normalization, quiz scoring, and the short-answer evaluator. ChromaDB and
PyMuPDF tests skip automatically if those packages are missing.

## YouTube transcripts

In the **Library** tab, open the **YouTube** tab, paste a YouTube URL and an
optional title, then click **Index transcript**. Supported URL shapes include
`youtube.com/watch?v=...`, `youtu.be/...`, Shorts, embed, live, and
`music.youtube.com` links. The video must have a public English transcript or
caption track. The loader requests common English variants such as `en`,
`en-US`, and `en-GB`; if none are available, the error lists the caption
languages the video does have.

The loader uses `youtube-transcript-api`, groups short caption snippets into
roughly one-minute pages, and keeps the timestamp of each page's first snippet.
Those timestamps are copied onto chunks and appear in retrieval previews, chat
sources, and quiz source references. Re-indexing the same transcript is
idempotent because the resource ID is based on the normalized transcript text.

## Chatting with your material

Open the **Chat** tab, choose the resources to search, and ask a question. The
chat uses the same resource-filtered retrieval and token-budgeted context builder
as quiz generation. It does not send your whole library to the model.

- Answers are grounded in retrieved passages. If nothing relevant is found, the
  chatbot says so instead of guessing.
- Each assistant answer has a **Sources** expander showing the source name, page
  number or YouTube timestamp, similarity score, and passage snippet.
- The chat keeps a short conversation history for follow-up questions. Use
  **Clear chat** to start over.
- Chat requires `LLM_API_KEY` and `LLM_MODEL`. Changing the resource selection
  affects new questions; previous answers stay in the transcript.

After generating a quiz, the **Quiz** tab asks whether you want to **Chat with
StudyMate first** or **Take the quiz now**. The Chat tab also has a **Return to
quiz** button while an unsubmitted quiz is active.

## Question generation

Open the **Generate** tab, choose resources and question types, set the total and
the easy/medium/hard counts, and optionally enter a topic. The counts must add up
exactly; otherwise the app shows a validation error and does not call the LLM.
Click **Generate questions**.

- With a topic, retrieval is restricted to the selected resources and ranked by
  cosine similarity.
- Without a topic, the app samples representative chunks across the selected
  resources so generation is not limited to one narrow passage.
- The LLM returns JSON, which is parsed with Pydantic. Invalid output, wrong
  counts, duplicates, missing source references, or missing selected types are
  sent back for up to two repair attempts. If generation still fails, the app
  reports an error instead of inventing questions.
- When generation succeeds, the full quiz (including the answer key) is kept in
  a process-local vault, and the app switches to the **Quiz** tab. Only a
  public view with question text, type/difficulty, and MCQ options is kept in
  session state. Before the quiz starts, you can choose to chat with the
  material first or begin immediately.
- The answer key is never rendered before submission. After submission, the app
  scores the attempt and reveals the correct answers, explanations, and source
  references in the review.

## Taking a quiz

Open the **Quiz** tab after generation. Before the first question, choose
**Chat with StudyMate first** or **Take the quiz now**. Once you start, the quiz
shows one question at a time with type and difficulty badges.

- Use **Previous**, **Next**, or **Jump to question** to move around. Answers save
  automatically, and the progress bar tracks how many questions are answered.
- Click **Submit quiz**. If any questions are unanswered, the app lists their
  numbers and asks for confirmation before submitting.
- After submission, the quiz is read-only and the app scores it. MCQ and
  fill-in-the-blank questions are exact-match; short answers are graded
  approximately by the LLM. The results summary shows total, correct, partial,
  incorrect, unanswered, score, and percentage.
- Review every question with your answer, the correct answer, the explanation,
  and source references. Short-answer cards label the AI grade as approximate.
  If some short answers could not be graded, they are excluded from the score and
  you can retry grading.
- **Start a new quiz** clears the active quiz and returns to the Generate tab.
- Active quizzes live in server memory. If the app restarts, generate a new
  quiz.

## Scoring and results

- **Multiple choice:** the selected option index is compared to the stored key. One point
  for a match.
- **Fill in the blank:** the answer is normalized with Unicode NFKC, casefolding,
  whitespace collapsing, dash normalization, and removal of edge punctuation/wrapping
  quotes. It matches the key or any configured accepted alternative. One point.
- **Short answer:** one LLM call grades the answer against the reference answer and key
  points on a 0–2 scale with a short justification. The grade maps to 0.0, 0.5, or 1.0
  points. The review labels this as approximate AI grading.
- **Unanswered:** zero points and counted separately as unanswered.
- **Ungraded short answers:** if the LLM is not configured or a grading call fails, the
  answer is marked not graded and excluded from the score denominator. Deterministic
  questions still score normally.
- The results summary shows the score and percentage plus correct, partial, incorrect,
  and unanswered counts. The review shows the student answer, correct answer,
  explanation, key points, and source references.

## Retrieval preview

In the **Generate** tab, expand **🔎 Retrieval preview (development tool)**. Choose
the resources to search, enter a topic or question, and click **Preview
retrieval**. The app shows the ranked passages with cosine similarity scores and
the formatted, token-budgeted context that chat and question generation receive.
No LLM call is made.

## Project layout

```
app.py                     Streamlit UI (library, chat, generation, quiz, tools)
.streamlit/config.toml     Streamlit theme and client defaults
Dockerfile                 CPU-only container image
docker-compose.yml         Compose service with a persistent data volume
.dockerignore              Keeps secrets, local data, and caches out of the image
DOCKER.md                  Docker setup, sharing, and troubleshooting guide
config/settings.py         Env-driven, validated settings
chat/
  schemas.py               Chat messages and cited source models
  service.py               Grounded retrieval + context + LLM orchestration
generation/
  llm_client.py            Groq client: complete, stream, connection check
  schemas.py               GenerationRequest, Question, GeneratedQuiz
  prompts.py               Grounding, JSON-schema, batch, and repair prompts
  validator.py             JSON parsing + type/count/source/duplicate validation
  question_generator.py    Retrieval/sampling, batching, repair loop, final quiz
quiz/
  schemas.py               Public QuizView/QuizQuestionView/QuizAttempt + post-submit results
  quiz_service.py          Key-stripping view conversion + process-local QuizVault
  evaluator.py             MCQ/FIB scoring + LLM short-answer grading orchestration
  scoring.py               Pure normalization, points, and percentage helpers
ingestion/
  pdf_loader.py            PyMuPDF extraction, one page per chunk source
  text_loader.py           Pasted notes
  youtube_loader.py        URL parsing + timestamped transcript pages
  normalization.py         Whitespace/artefact cleanup, content-based IDs
  chunker.py               Overlapping token windows + metadata
  indexer.py               Idempotent embed-and-store orchestration
  schemas.py               DocumentChunk, ResourceInfo
rag/
  embeddings.py            Lazy local SentenceTransformer embedder
  schemas.py               ContextPassage + RetrievedChunk with provenance/similarity
  vector_store.py          ChromaDB wrapper (upsert/query/sample/delete/list)
  retriever.py             Resource-filtered top_k retrieval + representative sampling
  context_builder.py       Dedupe, source labels, token-budget trimming
utils/                     Errors and logging
tests/                     Chunker, ingestion, YouTube, indexer, vector-store,
                           retrieval/context, chat, generation/validation,
                           quiz/scoring/evaluator, and LLM tests
```

## Known limitations

- Scanned/image-only PDFs are not OCR'd; the app explains this instead of
  silently indexing an empty document.
- YouTube ingestion needs a public transcript/caption track. The loader requests
  common English variants (`en`, `en-US`, `en-GB`, and similar); videos without
  one show a friendly error that lists the available caption languages. Caption
  snippets are grouped into roughly one-minute pages, so timestamps are
  page-level rather than per-word.
- Grounded chat and short-answer grading require a configured Groq model. If it
  is unavailable, chat is disabled and short answers are marked not graded and
  excluded from the score; deterministic questions still score.
- Active quizzes live in a process-local vault, so restarting the Streamlit server
  expires them. Submitted quizzes and results are not persisted to SQLite yet.
- Generation and short-answer grading require a valid Groq key; the default test
  suite uses mocked LLM responses and makes no network calls.
- Single-user application with no accounts or authentication. The Docker setup
  is intended for local use; do not expose it publicly without a private tunnel
  or an authenticating reverse proxy.

## Documentation

- [`prd.txt`](prd.txt) — original product requirements (note: the file is
  truncated partway through section 8.3).
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — decisions, architecture, data
  models, phases, and acceptance criteria.
- [`DOCKER.md`](DOCKER.md) — Docker setup, data persistence, sharing the app with
  a friend, and troubleshooting.
