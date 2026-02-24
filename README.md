# PCOS — Ontology-Governed Knowledge Base Framework

**Domain-agnostic, temporally-aware, self-evolving knowledge base powered by LLMs.**

Define a domain via YAML and automatically get a self-maintaining knowledge base with structured extraction, contradiction detection, and safe self-evolution — no code changes required.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Domain Schema Layer (YAML)                   │
│   entity types • relations • scopes • staleness • prompts │
├──────────────────────────────────────────────────────────┤
│                    FastAPI REST API                       │
│  /chat  /events  /beliefs  /schema  /entities  /evolution │
├──────────────────────────────────────────────────────────┤
│                     Brain Orchestrator                    │
├────────┬──────────┬──────────┬──────────┬───────────────┤
│Ingestion│ Compiler │Retrieval │ Runtime  │  Evolution    │
│        │(schema-  │ Planner  │(Plan/Exec│  Sandbox      │
│        │ driven)  │          │ /Reflect)│(dual-pipeline)│
├────────┴──────────┴──────────┴──────────┴───────────────┤
│ TTM (per-type staleness) │ Identity Resolution           │
├──────────────────────────┬───────────────────────────────┤
│    Security / Redaction  │  Eval Harness / Benchmarks    │
├──────────────────────────┴───────────────────────────────┤
│                    Memory Stores                          │
│  Episodic │ Semantic │ Procedural │ Policy │ KG Graph    │
│  (SQLite  │ (SQLite  │ (SQLite)   │(SQLite)│ (NetworkX)  │
│  +Chroma) │ +provenance)│          │        │             │
└──────────────────────────────────────────────────────────┘
```

## Domain Schema

Define your domain in YAML — PCOS generates Pydantic models, extraction prompts, API endpoints, and UI forms automatically:

```yaml
name: my-domain
version: "1.0"
scopes: [global, team, project]

entity_types:
  - name: Service
    fields:
      - { name: name, type: str, required: true }
      - { name: status, type: str, enum: [active, degraded, down] }
      - { name: owner, type: str }
    staleness_days: 30

relation_types:
  - { name: depends_on, source: Service, target: Service }
```

Ships with 3 example schemas: `domain.yaml` (personal assistant, 16 types), `regulatory-compliance.yaml` (7 types), `software-architecture.yaml` (8 types).

## Memory Types

| Type | Purpose | Storage |
|------|---------|---------|
| **Episodic** | What happened (conversations, events, outcomes) | SQLite + ChromaDB vectors |
| **Semantic** | What is true/believed (facts with temporal metadata) | SQLite with provenance |
| **Procedural** | How to do things (skills, workflows, templates) | SQLite (versioned) |
| **Working** | What matters right now (active goals, session state) | SQLite (persisted) |
| **Policy** | What is allowed (permissions, safety rules) | SQLite |

## Core API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Full cognitive pipeline: ingest → compile → retrieve → think → respond |
| `/api/v1/events/ingest` | POST | Ingest a raw event (text, voice, image, JSON/CSV) |
| `/api/v1/memory/compile` | POST | Compile candidate facts from an event |
| `/api/v1/memory/validate` | POST | Accept/reject a candidate fact |
| `/api/v1/query` | POST | Query the world model |
| `/api/v1/beliefs` | GET | List all active beliefs |
| `/api/v1/beliefs/{id}/explain` | GET | Explain provenance of a belief |
| `/api/v1/beliefs/history/{name}` | GET | Get belief history for an entity |
| `/api/v1/beliefs/{id}` | PUT | Update a belief |
| `/api/v1/beliefs/{id}` | DELETE | Retract a belief |
| `/api/v1/schema` | GET | Get the active domain schema |
| `/api/v1/entities/{entity_type}` | GET | List entities of a type |
| `/api/v1/entities/{entity_type}` | POST | Create an entity |
| `/api/v1/entities/{entity_type}/{id}` | GET | Get a single entity |
| `/api/v1/entities/{entity_type}/{id}` | DELETE | Delete an entity |
| `/api/v1/evolution/propose` | POST | Propose a self-evolution change |
| `/api/v1/evolution/proposals` | GET | List evolution proposals |
| `/api/v1/evolution/{id}/validate` | POST | Validate a proposal |
| `/api/v1/evolution/{id}/simulate` | POST | Simulate a proposal (dual-pipeline) |
| `/api/v1/evolution/{id}/approve` | POST | Approve a proposal |
| `/api/v1/evolution/{id}/deploy` | POST | Deploy a proposal |
| `/api/v1/evolution/{id}/rollback` | POST | Rollback a proposal |
| `/api/v1/maintenance/run` | POST | Run staleness detection & maintenance |
| `/api/v1/sync/export` | GET | Export data for cross-device sync |
| `/api/v1/sync/import` | POST | Import synced data with conflict resolution |
| `/api/v1/redact` | POST | Redact PII from text |
| `/api/v1/evaluation/metrics` | GET | Get evaluation metrics |
| `/api/v1/health` | GET | Health check |

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Run
percos
# or: uvicorn percos.app:app --reload

# 4. Test
pytest
```

## Design Principles

1. **Domain-agnostic by default** – define your domain in YAML, not code
2. **Schema as single source of truth** – extraction prompts, API endpoints, UI forms all generated from YAML
3. **Hybrid memory, not single-store** – different memory types for different needs
4. **Ontology as spine, not everything** – structured types ground the knowledge graph
5. **Time-aware truth management** – facts have lifetimes, per-type staleness from schema
6. **Provenance + confidence on all beliefs** – no untracked assertions
7. **Policy is first-class memory** – safety rules are stored and enforced
8. **Self-evolution must be sandboxed** – propose → validate → simulate → score → approve → deploy
9. **User can inspect and control memory** – view, edit, delete beliefs with provenance
10. **Everything important is versioned and auditable** – rollback support, audit trails

## Project Structure

```
domain.yaml                  # Default domain schema (personal assistant, 16 types)
schemas/
├── regulatory-compliance.yaml
└── software-architecture.yaml
src/percos/
├── __init__.py              # Package metadata
├── app.py                   # FastAPI application factory
├── cli.py                   # CLI entry point (serve, validate-schema)
├── config.py                # Settings (from .env)
├── llm.py                   # LLM client (OpenAI-compatible)
├── logging.py               # Structured logging
├── schema.py                # Domain schema loader (YAML → Pydantic models + prompts)
├── models/
│   ├── enums.py             # All enumerations (FactType, Confidence, etc.)
│   ├── ontology.py          # Base ontology entities
│   └── events.py            # Event, candidate, committed fact models
├── stores/
│   ├── database.py          # SQLAlchemy engine/session factory
│   ├── tables.py            # ORM table definitions
│   ├── episodic_store.py    # Episodic memory (SQL + ChromaDB)
│   ├── semantic_store.py    # Semantic memory (facts + provenance)
│   ├── procedural_policy_stores.py  # Procedural + policy stores
│   ├── graph.py             # In-memory knowledge graph (NetworkX)
│   └── audit_log.py         # Audit log store
├── engine/
│   ├── ingestion.py         # Multi-modal event ingestion
│   ├── compiler.py          # Schema-driven memory compiler
│   ├── ttm.py               # Temporal Truth Maintenance (per-type staleness)
│   ├── retrieval.py         # Hybrid retrieval planner (4 strategies)
│   ├── runtime.py           # Cognitive runtime (plan/execute/reflect)
│   ├── evolution.py         # Self-evolution sandbox (dual-pipeline simulation)
│   ├── evaluation.py        # Evaluation harness (P/R, conformance, agent metrics)
│   ├── consistency.py       # Consistency checker
│   ├── deadlines.py         # Deadline tracking
│   ├── identity_resolution.py  # Entity deduplication and merging
│   ├── integrations.py      # External integrations (Google Calendar, IMAP, MS Graph)
│   ├── security.py          # Redaction engine (PII detection + custom rules)
│   ├── style_tracker.py     # Communication style learning
│   ├── sync.py              # Cross-device sync (export/import)
│   └── brain.py             # Top-level orchestrator
└── api/
    ├── auth.py              # JWT authentication
    ├── deps.py              # FastAPI dependencies
    ├── schemas.py           # Request/response schemas
    ├── routes.py            # API route handlers
    └── panel.py             # Control panel UI (9 tabs, schema-driven)
```

## License

MIT
