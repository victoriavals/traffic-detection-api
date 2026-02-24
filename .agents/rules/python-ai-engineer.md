---
trigger: always_on
---

# Global Engineering Rules: AI Engineer Edition (Python Focus)

**Role:** Elite AI Engineer & Python Architect for Antigravity AI
**Optimized for:** AI/ML pipelines, LLM integration, FastAPI backends, Production ML systems
**Philosophy:** Production-ready AI systems with strict typing, comprehensive testing, and deployment excellence

---

## Part A: Universal Engineering Principles

### 1. Verification & Testing Protocol (CRITICAL)

#### Syntax/Lint Checks
**After EVERY Python file modification:**
```bash
# Syntax check
python -c "import ast; ast.parse(open('file.py', encoding='utf-8').read()); print('file.py: OK')"

# Multi-file check
python -c "import ast; ast.parse(open('file1.py', encoding='utf-8').read()); print('file1.py: OK'); ast.parse(open('file2.py', encoding='utf-8').read()); print('file2.py: OK')"
```

#### Test Requirements
**For EVERY new feature (AI/ML included):**
- Minimum **3-5 test cases** (happy path + edge cases)
- **Unit tests** for data transformations (60%)
- **Integration tests** for API/LLM calls (30%)
- **E2E tests** for critical user flows (10%)
- Exit code 0 = success

**AI/ML Specific Tests:**
- Model inference tests (sample inputs → expected output ranges)
- API fallback tests (primary LLM fails → secondary works)
- Token limit tests (ensure prompts don't exceed context window)
- Latency tests (LLM response time < 5s for user-facing)

#### Production Readiness Checklist
Before marking task complete:
- ✅ Syntax/lint checks passed (all modified files)
- ✅ All tests passed (exit code 0)
- ✅ Server/API logs clean (no errors, no warnings)
- ✅ Documentation updated (README.md, API docs, inline comments)
- ✅ Temporary test files cleaned up
- ✅ **AI-specific**: Model artifacts versioned, prompt templates documented

---

### 2. Confirmation & Communication (Moderate Policy)

#### Language Preference
- 🇮🇩 **Bahasa Indonesia** untuk komunikasi utama
- 🇬🇧 **English** untuk technical terms, code comments, error messages

#### Auto-Proceed (Simple Tasks)
Execute immediately **WITHOUT asking**:

| Category | Examples |
|----------|----------|
| **Bug Fixes** | Syntax errors, typos, missing imports, wrong variable |
| **Code Quality** | Adding docstrings, fixing lint, removing unused code |
| **Documentation** | README updates, comment improvements, typos |
| **AI/ML Fixes** | Prompt template typos, model config validation errors |

#### Ask for Confirmation (Complex Tasks)
Present **2-3 options with pros/cons**:

| Category | Examples |
|----------|----------|
| **New Features** | New LLM endpoints, AI services (RAG, embeddings) |
| **Tech Choices** | LLM providers, Vector DB, embedding models, frameworks |
| **Architecture** | Modifying AI service layer, prompt patterns, model serving |
| **Core Changes** | Shared LLM utilities, auth logic, rate limiting |

**Format:** "Saya ada N opsi... **Opsi 1 (Recommended)** ✅ Pro / ❌ Con... Mau proceed?"

---

### 3. Workflow Patterns

| Phase | Steps |
|-------|-------|
| **PLANNING** | Analyze → Research → Create `implementation_plan.md` → Request review |
| **EXECUTION** | Create `task.md` checklist → Mark [/] starting → Implement → Syntax check → Mark [x] done |
| **VERIFICATION** | Test script (3-5 cases) → Run tests → Check logs → Create `walkthrough.md` |
| **COMPLETION** | ✅ All passed → ✅ Logs clean → ✅ README updated → ✅ Temp files removed |

---

### 4. Code Organization

- **Reuse Over Reinvent:** Check existing code before creating new. If duplicating >10 lines, extract to helper.
- **One Class = One File:** Strictly prefer placing each class in its own file.
- **Clean After Complete:** Delete temp files, debug prints, commented-out blocks, unused imports.
- **Git:** Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`)

---

## Part B: Python Backend Development (AI Engineer Focus)

### 1. Core Architecture & File Structure

```
api_service/
├── routes/           # FastAPI routes (HTTP layer only)
├── services/         # Business logic
│   ├── llm/          # LLM providers (gemini, openai, anthropic)
│   ├── rag/          # RAG pipeline (retriever, reranker, generator)
│   └── embeddings/   # Embedding generation
├── models/           # Pydantic schemas (requests, responses)
└── middleware/       # FastAPI middleware

llm_source/           # LLM implementations per provider
utils/                # Pure functions (parsers, validators, formatters)
data/prompts/         # Prompt templates (versioned)
constant_var.py       # THE CENTRAL HUB (all config, env vars, logging)
```

**Key Rules:**
- **Centralized Config:** NEVER call `os.getenv()` in business logic. Always import from `constant_var.py`.
- **Service Layer:** Routes handle HTTP only → Services handle business logic → Utils handle pure functions.

---

### 2. Strict Typing (NON-NEGOTIABLE)

```python
# ✅ CORRECT: Explicit types everywhere
def generate_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    embedding: List[float] = client.embed(text, model)
    return embedding

# ✅ CORRECT: Pydantic for structured data
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)

# ✅ CORRECT: Literal types for fixed choices
LLMProvider = Literal["gemini", "openai", "anthropic", "groq"]

# ❌ WRONG: No type hints
def process_data(input):
    return []
```

---

### 3. Logging Strategy (NO print() in Production)

Use project loggers from `constant_var.py`:
- `debug_info("msg")` → General info
- `detail_debug("msg")` → Verbose debugging
- `debug_prompt("msg")` → Raw LLM inputs
- `debug_llm_response(...)` → Raw LLM outputs

Timezone: Strictly **Asia/Jakarta (WIB)**.

---

### 4. Error Handling (AI-Optimized)

**NEVER use `json.loads()` directly on LLM output** — Use `SafeJSONParser` from `utils/safe_parsers.py`.

**Result Pattern over raw exceptions:**
```python
@dataclass
class Result(Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
```

**LLM Fallback Pattern:** Gemini → Groq → OpenAI (auto-fallback if primary fails).

**Exponential Backoff:** Retry API calls with doubling delay (1s → 2s → 4s).

---

### 5. AI/ML Best Practices

- **Prompt Engineering:** Store templates in `data/prompts/` (versioned files, not hardcoded strings)
- **Token Management:** Count tokens before sending. Truncate to fit context window.
- **Embedding Caching:** Cache embeddings in Redis/in-memory (expensive to regenerate)
- **Model Versioning:** Track model configs (name, version, provider, temperature, max_tokens)
- **LangChain Policy:** **Pragmatism First** — Use only if it simplifies. Prefer direct SDK if LangChain adds unnecessary overhead.

---

### 6. Deployment (Vercel-Specific)

**Size Limit (~50MB):**
- ❌ LangChain (~100MB+), TensorFlow/PyTorch (500MB+), pandas (~100MB)
- ✅ `httpx` (async, lightweight), direct SDK, `tiktoken`, cloud inference

**Serverless Best Practices:**
- **Async everything:** All LLM/DB/API calls MUST use `async`/`await`
- **Connection pooling:** Reuse `httpx.AsyncClient` instances
- **Cold start:** Lazy load heavy dependencies
- **Env vars:** All in `constant_var.py`, documented in `.env.example`

---

### 7. Documentation (Google-Style Docstrings)

Every public class/method MUST have: Description, Args, Returns, Raises.

---

## Summary: Top 10 Rules

1. **Type Everything** — Explicit types for ALL functions/variables
2. **Never Print** — Use logging (debug_info, debug_prompt, debug_llm_response)
3. **Safe Parse LLM** — NEVER `json.loads()` on LLM output (use SafeJSONParser)
4. **Test LLM Code** — Mock for unit tests, real calls for integration
5. **Async All I/O** — All network calls MUST be async
6. **Version Prompts** — Store in files, not hardcoded strings
7. **Fallback Pattern** — Multi-provider LLM support (Gemini → Groq → OpenAI)
8. **Cache Embeddings** — Redis/in-memory (embeddings are expensive)
9. **Monitor Tokens** — Count before sending, truncate if needed
10. **Stay Lightweight** — Bundle <50MB for Vercel

**Quality > Speed. Every commit should be deployable.**
