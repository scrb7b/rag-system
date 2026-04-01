# RAG Q&A System

Pipeline Retrieval-Augmented Generation для відповідей на запитання за документами `.pdf` та `.html`.

Система індексує документи, будує гібридний (dense + sparse BM25) векторний пошук і генерує відповіді через LLM — виключно на основі знайденого контексту.

## Стек

| Компонент | Технологія                                                                  |
|---|-----------------------------------------------------------------------------|
| Завантаження та парсинг документів | [Docling](https://github.com/DS4SD/docling) (PDF + HTML)                    |
| Розбивка на чанки | `MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter` (langchain_text_splitters)                  |
| Векторне сховище | [Qdrant](https://qdrant.tech/) (in-memory, hybrid dense+sparse)             |
| Dense embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` via fastembed |
| Sparse embeddings | `Qdrant/bm25`                                                               |
| Reranker (опціонально) | `jinaai/jina-reranker-v2-base-multilingual` via fastembed                   |
| LLM | [Ollama](https://ollama.com/) (локально) або OpenAI API                     |
| Логування | `structlog`                                                                 |
| Тести | `pytest` (32 тести)                                                         |
| Бенчмарк | [RAGAS](https://docs.ragas.io/)                                             |

## Вимоги

- Python 3.12
- [Ollama](https://ollama.com/) — якщо використовується локальний LLM

## Встановлення

### Варіант A — uv (рекомендовано)

```bash
# Встановити uv (якщо ще немає)
curl -Ls https://astral.sh/uv/install.sh | sh

git clone <repo-url>
cd kernel-rag

uv sync
cp src/.env.example src/.env
```

### Варіант B — pip + venv

```bash
git clone <repo-url>
cd kernel-rag

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.ps1

pip install -e .
cp src/.env.example src/.env
```

## Налаштування (`src/.env`)

```env
# Провайдер LLM: "ollama" (локально) або "openai"
LLM_PROVIDER=ollama

# Якщо ollama:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Якщо openai:
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-3.5-turbo

# Модель ембеддингів (завантажується автоматично при першому запуску)
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
SPARSE_EMBED_MODEL=Qdrant/bm25

# Параметри пошуку
TOP_K=10
ENABLE_RERANKING=true   # рекомендовано, модель ~580MB завантажується один раз
SCORE_THRESHOLD=0.0
```

Повний список змінних — у [`src/.env.example`](src/.env.example).

## Запуск

### Передумова: покласти документи в `./data`

```
data/
├── document.pdf
└── page.html
```

Підтримуються `.pdf`, `.html`, `.htm` — можна вкладені папки.

### Jupyter Notebook (рекомендовано для демо)

```bash
# uv
uv run jupyter notebook demo.ipynb

# pip
jupyter notebook demo.ipynb
```

Ноутбук проходить весь pipeline покроково: завантаження документів → індексація → тест пошуку → Q&A агент → інтерактивний режим.

### CLI з Ollama (локальний LLM)

```bash
# 1. Запустити Ollama та завантажити модель (один раз)
ollama pull llama3.2

# 2. Запустити pipeline
# uv
uv run python -m src.main
# pip
python -m src.main
```

### CLI з OpenAI

```bash
# У src/.env:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...

# uv
uv run python -m src.main
# pip
python -m src.main
```

### Приклад сесії

```
Chat is ready. Type your question or 'exit' to quit.

Question: Яка молекулярна маса гліфосату?

Молекулярна маса гліфосату становить 169,1.

Sources: Гліфосат.html (0.87)
```

## Архітектура

```
kernel-rag/
├── data/                        # Документи для індексації (.pdf, .html)
├── src/
│   ├── main.py                  # Точка входу (CLI чат)
│   ├── agent.py                 # QAAgent: пошук → генерація → QAResult
│   ├── config.py                # Налаштування через .env (Pydantic Settings)
│   ├── logging_setup.py         # Конфігурація structlog
│   ├── ingestion/
│   │   ├── loaders.py           # PDF/HTML → список чанків (через Docling)
│   │   └── embedder.py          # VectorStore (Qdrant hybrid + reranker)
│   ├── generation/
│   │   ├── base.py              # BaseLLM: prompt builder + абстракція
│   │   ├── openai_llm.py        # OpenAI реалізація
│   │   └── ollama_llm.py        # Ollama реалізація
│   └── benchmark/
│       ├── bench_ragas.py       # RAGAS оцінка якості
│       ├── utils.py             # Допоміжні функції для бенчмарку
│       └── samples.py           # Тестові Q&A пари
├── tests/                       # pytest тести
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_embedder.py
│   ├── test_generation.py
│   └── test_loaders.py
├── demo.ipynb                   # Jupyter демо
├── pyproject.toml
└── src/.env                     # Конфіг (не в git)
```

### Як працює pipeline

```
PDF/HTML
   │
   ▼
Docling → HierarchicalChunker → чанки тексту
   │
   ▼
fastembed  (dense: mpnet-multilingual  +  sparse: BM25)
   │
   ▼
Qdrant in-memory  (hybrid search)
   │
   ▼  (якщо ENABLE_RERANKING=true)
jina-reranker-v2-base-multilingual → пересортування
   │
   ▼
TOP_K чанків → промпт → Ollama / OpenAI
   │
   ▼
Відповідь виключно на основі контексту
```

## Тести

```bash
# uv
uv run pytest tests/ -v

# pip
pytest tests/ -v
```

32 тести покривають: VectorStore (пошук, індексація, скидання), QAAgent (відповіді, fallback, джерела), генерацію (промпт, контекст), завантажувачі (формати, помилки).

## Бенчмарк (RAGAS)

Оцінює якість RAG за 4 метриками: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

```bash
# uv
uv run python -m src.benchmark.bench_ragas --provider ollama

# pip
python -m src.benchmark.bench_ragas --provider ollama

# З OpenAI (більш надійний judge)
python -m src.benchmark.bench_ragas --provider openai

# Зберегти результати
python -m src.benchmark.bench_ragas --output results.json
```

> **Важливо:** RAGAS використовує LLM як judge. Для Ollama потрібна модель з підтримкою OpenAI-compatible API — Ollama підтримує це з версії 0.1.24.

### Результати (llama3.2 + llm3.2-as-judge, 6 зразків)

| Метрика | Значення | Опис |
|---|---|---|
| `faithfulness` | **0.6250** | Відповідь спирається на контекст (без галюцинацій) |
| `answer_relevancy` | **0.7521** | Відповідь відповідає запитанню |
| `context_precision` | n/a | Вимагає reference-відповідей у тестовому наборі |
| `context_recall` | **0.7917** | Контекст покриває необхідну інформацію |

> `context_precision = nan` — метрика потребує поля `reference` у тестових зразках (`samples.py`).

## Підтримувані моделі ембеддингів

| Модель | Розмірність | Розмір | Якість |
|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | ~420 MB | добре |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~120 MB | базово |
| `intfloat/multilingual-e5-large` | 1024 | ~1.1 GB | відмінно |

При зміні моделі потрібно переіндексувати документи (перезапустити pipeline).
