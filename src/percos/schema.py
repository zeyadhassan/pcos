"""Domain Schema System — the foundation of the domain-agnostic framework.

Loads a YAML domain schema and provides:
  1. Dynamic Pydantic model generation via ``pydantic.create_model()``
  2. Auto-generated extraction prompts for the Memory Compiler
  3. Entity / relation type registries for compiler, retrieval, TTM, and KG
  4. Per-entity-type staleness configuration
  5. Configurable scopes
  6. Schema validation and error reporting
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model

from percos.logging import get_logger

log = get_logger("schema")

# ── Field type mapping from YAML → Python ───────────────
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "str": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "datetime": datetime,
    "date": datetime,
    "list": list,
    "dict": dict,
}

# Default staleness for entity types that don't specify one
DEFAULT_STALENESS_DAYS = 90


# ── Data containers ─────────────────────────────────────

class FieldDef(BaseModel):
    """Definition of a single field in an entity type."""
    name: str
    type: str = "string"
    required: bool = False
    enum: list[str] | None = None
    description: str = ""


class EntityTypeDef(BaseModel):
    """Definition of an entity type from the domain schema."""
    name: str
    fields: list[FieldDef] = Field(default_factory=list)
    staleness_days: int = DEFAULT_STALENESS_DAYS
    description: str = ""


class RelationTypeDef(BaseModel):
    """Definition of a relation type from the domain schema."""
    name: str
    from_type: str
    to_type: str
    description: str = ""


class DomainSchema(BaseModel):
    """Parsed domain schema with all definitions and generated artefacts."""
    name: str = "default"
    version: str = "1.0"
    description: str = ""
    scopes: list[str] = Field(default_factory=lambda: ["global"])
    entity_types: dict[str, EntityTypeDef] = Field(default_factory=dict)
    relation_types: dict[str, RelationTypeDef] = Field(default_factory=dict)

    # Runtime artefacts (not serialised to YAML)
    _pydantic_models: dict[str, type[BaseModel]] = {}
    _extraction_prompt: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Public API ──────────────────────────────────────

    def get_entity_type_names(self) -> list[str]:
        """Return the list of entity type names (lowercase)."""
        return [n.lower() for n in self.entity_types]

    def get_staleness_days(self, entity_type: str) -> int:
        """Return staleness_days for the given entity type, or default."""
        for name, et in self.entity_types.items():
            if name.lower() == entity_type.lower():
                return et.staleness_days
        return DEFAULT_STALENESS_DAYS

    def get_scopes(self) -> list[str]:
        """Return the configured scopes for this domain."""
        return self.scopes

    def get_pydantic_model(self, entity_type: str) -> type[BaseModel] | None:
        """Return the dynamic Pydantic model for the given entity type."""
        return self._pydantic_models.get(entity_type.lower())

    def get_model_map(self) -> dict[str, type[BaseModel]]:
        """Return the full entity_type → Pydantic model mapping."""
        return dict(self._pydantic_models)

    def get_extraction_prompt(self) -> str:
        """Return the auto-generated extraction system prompt."""
        return self._extraction_prompt

    def get_relation_type_names(self) -> list[str]:
        """Return the list of relation type names."""
        return list(self.relation_types.keys())

    def validate_scope(self, scope: str) -> bool:
        """Check whether a scope value is valid for this domain."""
        return scope.lower() in [s.lower() for s in self.scopes]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the schema to a dict (for the /schema endpoint)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "scopes": self.scopes,
            "entity_types": {
                name: {
                    "fields": [f.model_dump() for f in et.fields],
                    "staleness_days": et.staleness_days,
                    "description": et.description,
                }
                for name, et in self.entity_types.items()
            },
            "relation_types": {
                name: {
                    "from": rt.from_type,
                    "to": rt.to_type,
                    "description": rt.description,
                }
                for name, rt in self.relation_types.items()
            },
        }


# ── Loading & parsing ───────────────────────────────────

def load_domain_schema(path: str | Path) -> DomainSchema:
    """Load and parse a domain schema YAML file.

    Returns a ``DomainSchema`` with generated Pydantic models and extraction prompt.

    Raises ``SchemaValidationError`` on invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise SchemaValidationError(f"Domain schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise SchemaValidationError("Domain schema must be a YAML mapping at the top level")

    return _parse_schema(raw, source_path=str(path))


def parse_schema_dict(raw: dict[str, Any]) -> DomainSchema:
    """Parse a schema from a plain dict (useful for tests / programmatic use)."""
    if not isinstance(raw, dict):
        raise SchemaValidationError("Schema input must be a YAML mapping (dict), got " + type(raw).__name__)
    return _parse_schema(raw, source_path="<dict>")


def _parse_schema(raw: dict[str, Any], *, source_path: str) -> DomainSchema:
    """Internal: parse raw dict → ``DomainSchema`` with validation."""
    errors: list[str] = []

    name = raw.get("name", "default")
    version = str(raw.get("version", "1.0"))
    description = raw.get("description", "")
    scopes = raw.get("scopes", ["global"])

    if not isinstance(scopes, list) or not scopes:
        errors.append("'scopes' must be a non-empty list")
        scopes = ["global"]

    # ── Parse entity types ──────────────────────────────
    entity_types: dict[str, EntityTypeDef] = {}
    raw_entities = raw.get("entity_types", {})
    if not isinstance(raw_entities, dict):
        errors.append("'entity_types' must be a mapping")
        raw_entities = {}

    seen_entity_names: set[str] = set()
    for et_name, et_def in raw_entities.items():
        lower_name = et_name.lower()
        if lower_name in seen_entity_names:
            errors.append(f"Duplicate entity type (case-insensitive): '{et_name}'")
            continue
        seen_entity_names.add(lower_name)

        if not isinstance(et_def, dict):
            errors.append(f"Entity type '{et_name}' must be a mapping")
            continue

        fields: list[FieldDef] = []
        raw_fields = et_def.get("fields", {})
        if isinstance(raw_fields, dict):
            for fname, fdef in raw_fields.items():
                if isinstance(fdef, dict):
                    ftype = fdef.get("type", "string")
                    if ftype not in _TYPE_MAP and ftype not in ("list", "dict"):
                        errors.append(
                            f"Entity '{et_name}', field '{fname}': unknown type '{ftype}'. "
                            f"Allowed: {', '.join(_TYPE_MAP.keys())}"
                        )
                    fields.append(FieldDef(
                        name=fname,
                        type=ftype,
                        required=fdef.get("required", False),
                        enum=fdef.get("enum"),
                        description=fdef.get("description", ""),
                    ))
                else:
                    # Short form: field_name: type
                    fields.append(FieldDef(name=fname, type=str(fdef)))

        staleness = et_def.get("staleness_days", DEFAULT_STALENESS_DAYS)
        entity_types[et_name] = EntityTypeDef(
            name=et_name,
            fields=fields,
            staleness_days=int(staleness),
            description=et_def.get("description", ""),
        )

    # ── Parse relation types ────────────────────────────
    relation_types: dict[str, RelationTypeDef] = {}
    raw_relations = raw.get("relation_types", {})
    if isinstance(raw_relations, dict):
        for rname, rdef in raw_relations.items():
            if not isinstance(rdef, dict):
                errors.append(f"Relation type '{rname}' must be a mapping with 'from' and 'to'")
                continue
            from_type = rdef.get("from", "")
            to_type = rdef.get("to", "")
            if from_type and from_type not in raw_entities:
                errors.append(f"Relation '{rname}': 'from' type '{from_type}' not in entity_types")
            if to_type and to_type not in raw_entities:
                errors.append(f"Relation '{rname}': 'to' type '{to_type}' not in entity_types")
            relation_types[rname] = RelationTypeDef(
                name=rname,
                from_type=from_type,
                to_type=to_type,
                description=rdef.get("description", ""),
            )

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed ({source_path}):\n  - " + "\n  - ".join(errors)
        )

    # ── Build schema object ─────────────────────────────
    schema = DomainSchema(
        name=name,
        version=version,
        description=description,
        scopes=scopes,
        entity_types=entity_types,
        relation_types=relation_types,
    )

    # ── Generate dynamic Pydantic models ────────────────
    schema._pydantic_models = _build_pydantic_models(entity_types)

    # ── Generate extraction prompt ──────────────────────
    schema._extraction_prompt = _build_extraction_prompt(schema)

    log.info(
        "schema_loaded",
        name=name,
        version=version,
        entity_types=len(entity_types),
        relation_types=len(relation_types),
        scopes=scopes,
    )

    return schema


# ── Dynamic model generation ────────────────────────────

def _build_pydantic_models(entity_types: dict[str, EntityTypeDef]) -> dict[str, type[BaseModel]]:
    """Generate Pydantic models from entity type definitions.

    Model keys are **lowercase** entity type names for case-insensitive lookup.
    """
    models: dict[str, type[BaseModel]] = {}

    for et_name, et_def in entity_types.items():
        field_definitions: dict[str, Any] = {}

        for f in et_def.fields:
            python_type = _TYPE_MAP.get(f.type, str)
            default = ... if f.required else None
            # For optional fields, wrap type in Optional
            if f.required:
                field_definitions[f.name] = (python_type, Field(description=f.description or f.name))
            else:
                field_definitions[f.name] = (
                    python_type | None,
                    Field(default=default, description=f.description or f.name),
                )

        model = create_model(
            et_name,
            **field_definitions,
        )
        models[et_name.lower()] = model

    return models


# ── Extraction prompt generation ────────────────────────

def _build_extraction_prompt(schema: DomainSchema) -> str:
    """Auto-generate the extraction system prompt from the domain schema."""

    entity_descriptions: list[str] = []
    for et_name, et_def in schema.entity_types.items():
        fields_desc = []
        for f in et_def.fields:
            parts = [f.name]
            if f.type != "string":
                parts.append(f"({f.type})")
            if f.required:
                parts.append("[required]")
            if f.enum:
                parts.append(f"[one of: {', '.join(f.enum)}]")
            fields_desc.append(" ".join(parts))
        entity_descriptions.append(
            f"  - {et_name}: {', '.join(fields_desc)}"
        )

    entity_type_names = [n.lower() for n in schema.entity_types.keys()]
    scopes_str = ", ".join(schema.scopes)

    relation_descriptions: list[str] = []
    for rname, rdef in schema.relation_types.items():
        relation_descriptions.append(f"  - {rname}: {rdef.from_type} → {rdef.to_type}")

    relations_section = ""
    if relation_descriptions:
        relations_section = (
            "\n\nKnown relation types:\n"
            + "\n".join(relation_descriptions)
            + "\n\nIf the input establishes a relation between entities, include "
            "relation_type, relation_source (name of source entity), and "
            "relation_target (name of target entity)."
        )

    prompt = f"""\
You are a knowledge extraction engine for the "{schema.name}" domain.
Given a raw event (conversation, document, structured data, etc.),
extract structured knowledge candidates matching the domain schema.

Entity types and their fields:
{chr(10).join(entity_descriptions)}

For each candidate, output a JSON object with:
- entity_type: one of [{', '.join(entity_type_names)}]
- entity_data: dict with the fields defined above
- relation_type: optional string if this establishes a relation
- relation_source: optional name/id of source entity
- relation_target: optional name/id of target entity
- fact_type: one of [observed, derived, hypothesis, policy]
- confidence: one of [high, medium, low]
- scope: one of [{scopes_str}]
- sensitivity: one of [public, internal, private, secret]
{relations_section}

Return a JSON object with a "candidates" array.
If no knowledge can be extracted, return {{"candidates": []}}.
Be precise. Do not hallucinate entities. Only extract what is clearly present or
reasonably inferable from the input.
"""
    return prompt


# ── Schema validation error ─────────────────────────────

class SchemaValidationError(Exception):
    """Raised when a domain schema YAML is invalid."""
    pass


# ── Singleton schema management ─────────────────────────

_active_schema: DomainSchema | None = None


def get_domain_schema() -> DomainSchema:
    """Return the active domain schema (loaded on first call).

    Uses the ``domain_schema_path`` setting from config.
    """
    global _active_schema
    if _active_schema is not None:
        return _active_schema

    from percos.config import get_settings
    settings = get_settings()
    schema_path = getattr(settings, "domain_schema_path", "./domain.yaml")
    path = Path(schema_path)

    if not path.exists():
        log.warning("schema_not_found_using_default", path=str(path))
        _active_schema = _build_default_schema()
        return _active_schema

    _active_schema = load_domain_schema(path)
    return _active_schema


def set_domain_schema(schema: DomainSchema) -> None:
    """Override the active domain schema (useful for tests)."""
    global _active_schema
    _active_schema = schema


def reset_domain_schema() -> None:
    """Clear the cached schema so it gets reloaded on next access."""
    global _active_schema
    _active_schema = None


def _build_default_schema() -> DomainSchema:
    """Build a minimal default schema when no YAML is provided."""
    raw = {
        "name": "default",
        "version": "1.0",
        "description": "Default schema — accepts any entity type",
        "scopes": ["global", "work", "personal", "project"],
        "entity_types": {
            "Entity": {
                "fields": {
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "staleness_days": DEFAULT_STALENESS_DAYS,
            },
        },
        "relation_types": {},
    }
    return parse_schema_dict(raw)
