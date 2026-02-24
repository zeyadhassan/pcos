"""Tests for the Domain Schema System (§4.A, §5, §14) — GAP-7.

Covers:
- load_domain_schema() with valid and invalid YAML
- _build_pydantic_models() dynamic model generation
- _build_extraction_prompt() prompt generation
- DomainSchema methods: get_entity_type_names, get_staleness_days,
  get_scopes, validate_scope, get_model_map, to_dict, etc.
- Schema serialisation round-trip
- Multi-domain loading (regulatory-compliance, software-architecture)
- Framework base types (Entity, FactMetadata, Relation) preservation
- Invalid YAML error handling
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from percos.schema import (
    DEFAULT_STALENESS_DAYS,
    DomainSchema,
    SchemaValidationError,
    _build_extraction_prompt,
    _build_pydantic_models,
    load_domain_schema,
    parse_schema_dict,
    reset_domain_schema,
    set_domain_schema,
)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_YAML = PROJECT_ROOT / "domain.yaml"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"


# ── Helpers ─────────────────────────────────────────────

MINIMAL_SCHEMA = {
    "name": "test-domain",
    "version": "2.0",
    "description": "A test domain schema",
    "scopes": ["global", "department"],
    "entity_types": {
        "Widget": {
            "fields": {
                "name": {"type": "string", "required": True},
                "colour": {"type": "string"},
                "weight": {"type": "float"},
                "active": {"type": "bool"},
            },
            "staleness_days": 45,
            "description": "A test widget entity",
        },
        "Gadget": {
            "fields": {
                "name": {"type": "string", "required": True},
                "status": {"type": "string", "enum": ["new", "used", "broken"]},
                "created": {"type": "datetime"},
            },
        },
    },
    "relation_types": {
        "contains": {
            "from": "Widget",
            "to": "Gadget",
            "description": "A widget contains a gadget",
        },
    },
}


# ── Tests: parse_schema_dict / load_domain_schema ──────

class TestParseSchemaDict:
    """Test parsing schemas from plain dicts."""

    def test_minimal_schema(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.name == "test-domain"
        assert schema.version == "2.0"
        assert schema.description == "A test domain schema"
        assert schema.scopes == ["global", "department"]
        assert len(schema.entity_types) == 2
        assert "Widget" in schema.entity_types
        assert "Gadget" in schema.entity_types
        assert len(schema.relation_types) == 1
        assert "contains" in schema.relation_types

    def test_entity_type_names(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        names = schema.get_entity_type_names()
        assert "widget" in names
        assert "gadget" in names
        assert len(names) == 2

    def test_staleness_days(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.get_staleness_days("Widget") == 45
        assert schema.get_staleness_days("Gadget") == DEFAULT_STALENESS_DAYS
        assert schema.get_staleness_days("NonExistent") == DEFAULT_STALENESS_DAYS

    def test_staleness_days_case_insensitive(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.get_staleness_days("widget") == 45
        assert schema.get_staleness_days("WIDGET") == 45

    def test_scopes(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.get_scopes() == ["global", "department"]

    def test_validate_scope(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.validate_scope("global") is True
        assert schema.validate_scope("department") is True
        assert schema.validate_scope("GLOBAL") is True  # case-insensitive
        assert schema.validate_scope("personal") is False
        assert schema.validate_scope("nonexistent") is False

    def test_relation_types(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        names = schema.get_relation_type_names()
        assert "contains" in names
        rt = schema.relation_types["contains"]
        assert rt.from_type == "Widget"
        assert rt.to_type == "Gadget"
        assert rt.description == "A widget contains a gadget"


class TestDynamicModelGeneration:
    """Test Pydantic model creation from entity type definitions."""

    def test_models_generated(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        model_map = schema.get_model_map()
        assert "widget" in model_map
        assert "gadget" in model_map

    def test_model_fields(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        WidgetModel = schema.get_pydantic_model("widget")
        assert WidgetModel is not None
        fields = WidgetModel.model_fields
        assert "name" in fields
        assert "colour" in fields
        assert "weight" in fields
        assert "active" in fields

    def test_required_field(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        WidgetModel = schema.get_pydantic_model("widget")
        assert WidgetModel is not None
        # Required field should raise on missing
        with pytest.raises(Exception):
            WidgetModel()  # missing required 'name'

    def test_optional_field_defaults(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        WidgetModel = schema.get_pydantic_model("widget")
        assert WidgetModel is not None
        instance = WidgetModel(name="test")
        assert instance.name == "test"
        assert instance.colour is None
        assert instance.weight is None
        assert instance.active is None

    def test_model_case_insensitive_lookup(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.get_pydantic_model("Widget") is not None
        assert schema.get_pydantic_model("WIDGET") is not None
        assert schema.get_pydantic_model("widget") is not None

    def test_model_nonexistent(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        assert schema.get_pydantic_model("NonExistent") is None


class TestExtractionPrompt:
    """Test auto-generated extraction prompt from schema."""

    def test_prompt_generated(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert len(prompt) > 0

    def test_prompt_contains_entity_types(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "widget" in prompt.lower()
        assert "gadget" in prompt.lower()

    def test_prompt_contains_fields(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "name" in prompt
        assert "colour" in prompt
        assert "weight" in prompt

    def test_prompt_contains_scopes(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "global" in prompt
        assert "department" in prompt

    def test_prompt_contains_enum_values(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "new" in prompt
        assert "used" in prompt
        assert "broken" in prompt

    def test_prompt_contains_relation_types(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "contains" in prompt
        assert "Widget" in prompt
        assert "Gadget" in prompt

    def test_prompt_mentions_domain_name(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        prompt = schema.get_extraction_prompt()
        assert "test-domain" in prompt


class TestSerialization:
    """Test schema to_dict serialisation."""

    def test_to_dict_round_trip(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        d = schema.to_dict()
        assert d["name"] == "test-domain"
        assert d["version"] == "2.0"
        assert len(d["entity_types"]) == 2
        assert len(d["relation_types"]) == 1

    def test_to_dict_fields(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        d = schema.to_dict()
        widget = d["entity_types"]["Widget"]
        assert "fields" in widget
        field_names = [f["name"] for f in widget["fields"]]
        assert "name" in field_names
        assert "colour" in field_names

    def test_to_dict_relation_types(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        d = schema.to_dict()
        contains = d["relation_types"]["contains"]
        assert contains["from"] == "Widget"
        assert contains["to"] == "Gadget"


class TestInvalidSchemas:
    """Test error handling for invalid YAML schemas."""

    def test_non_dict_raises(self):
        with pytest.raises(SchemaValidationError, match="YAML mapping"):
            parse_schema_dict("not a dict")  # type: ignore

    def test_duplicate_entity_type_raises(self):
        raw = {
            "entity_types": {
                "Widget": {"fields": {"name": "string"}},
                "widget": {"fields": {"name": "string"}},  # case-insensitive dupe
            },
        }
        with pytest.raises(SchemaValidationError, match="Duplicate"):
            parse_schema_dict(raw)

    def test_unknown_field_type(self):
        raw = {
            "entity_types": {
                "Foo": {
                    "fields": {
                        "bar": {"type": "uuid_exotic"},
                    },
                },
            },
        }
        with pytest.raises(SchemaValidationError, match="unknown type"):
            parse_schema_dict(raw)

    def test_invalid_relation_from_type(self):
        raw = {
            "entity_types": {
                "Widget": {"fields": {"name": "string"}},
            },
            "relation_types": {
                "uses": {"from": "NonExistent", "to": "Widget"},
            },
        }
        with pytest.raises(SchemaValidationError, match="not in entity_types"):
            parse_schema_dict(raw)

    def test_file_not_found(self):
        with pytest.raises(SchemaValidationError, match="not found"):
            load_domain_schema("/nonexistent/path.yaml")

    def test_empty_scopes_corrected(self):
        raw = {
            "scopes": [],
            "entity_types": {
                "Foo": {"fields": {"name": "string"}},
            },
        }
        with pytest.raises(SchemaValidationError, match="non-empty"):
            parse_schema_dict(raw)

    def test_entity_type_not_a_mapping(self):
        raw = {
            "entity_types": {
                "Foo": "not a mapping",
            },
        }
        with pytest.raises(SchemaValidationError, match="must be a mapping"):
            parse_schema_dict(raw)


class TestDefaultSchema:
    """Test default schema fallback."""

    def test_default_schema_builds(self):
        from percos.schema import _build_default_schema
        schema = _build_default_schema()
        assert schema.name == "default"
        assert "global" in schema.scopes
        assert len(schema.entity_types) >= 1

    def test_set_and_reset_schema(self):
        schema = parse_schema_dict(MINIMAL_SCHEMA)
        set_domain_schema(schema)
        from percos.schema import get_domain_schema
        assert get_domain_schema().name == "test-domain"
        reset_domain_schema()


class TestShortFormFields:
    """Test short-form field definition (field_name: type)."""

    def test_short_form(self):
        raw = {
            "entity_types": {
                "Simple": {
                    "fields": {
                        "name": "string",
                        "count": "int",
                    },
                },
            },
        }
        schema = parse_schema_dict(raw)
        et = schema.entity_types["Simple"]
        assert len(et.fields) == 2
        assert et.fields[0].type == "string"
        assert et.fields[1].type == "int"


# ── Tests: Multi-Domain Loading ────────────────────────

class TestLoadDomainYAML:
    """Test loading the main domain.yaml personal-assistant schema."""

    def test_load_domain_yaml(self):
        if not DOMAIN_YAML.exists():
            pytest.skip("domain.yaml not found at project root")
        schema = load_domain_schema(DOMAIN_YAML)
        assert schema.name == "personal-assistant"
        assert "global" in schema.scopes
        # Should have the 16 entity types
        assert len(schema.entity_types) >= 14

    def test_domain_yaml_entity_types(self):
        if not DOMAIN_YAML.exists():
            pytest.skip("domain.yaml not found at project root")
        schema = load_domain_schema(DOMAIN_YAML)
        names = schema.get_entity_type_names()
        for expected in ["person", "task", "project", "goal", "preference"]:
            assert expected in names, f"Expected entity type '{expected}' not found"

    def test_domain_yaml_models(self):
        if not DOMAIN_YAML.exists():
            pytest.skip("domain.yaml not found at project root")
        schema = load_domain_schema(DOMAIN_YAML)
        TaskModel = schema.get_pydantic_model("task")
        assert TaskModel is not None
        instance = TaskModel(name="Buy groceries")
        assert instance.name == "Buy groceries"

    def test_domain_yaml_staleness(self):
        if not DOMAIN_YAML.exists():
            pytest.skip("domain.yaml not found at project root")
        schema = load_domain_schema(DOMAIN_YAML)
        assert schema.get_staleness_days("Task") == 30
        assert schema.get_staleness_days("Person") == 365
        assert schema.get_staleness_days("CalendarEvent") == 1

    def test_domain_yaml_relation_types(self):
        if not DOMAIN_YAML.exists():
            pytest.skip("domain.yaml not found at project root")
        schema = load_domain_schema(DOMAIN_YAML)
        rels = schema.get_relation_type_names()
        assert "manages" in rels
        assert "assigned_to" in rels


class TestMultiDomainSchemas:
    """Test that non-personal-assistant domain schemas load correctly."""

    def test_regulatory_compliance_schema(self):
        path = SCHEMAS_DIR / "regulatory-compliance.yaml"
        if not path.exists():
            pytest.skip("regulatory-compliance.yaml not found")
        schema = load_domain_schema(path)
        assert schema.name == "regulatory-compliance"
        names = schema.get_entity_type_names()
        assert "regulation" in names
        assert "auditfinding" in names
        assert "correctiveaction" in names
        # Scopes should be domain-specific
        assert "regulation" in schema.get_scopes()
        assert "department" in schema.get_scopes()

    def test_regulatory_models(self):
        path = SCHEMAS_DIR / "regulatory-compliance.yaml"
        if not path.exists():
            pytest.skip("regulatory-compliance.yaml not found")
        schema = load_domain_schema(path)
        RegModel = schema.get_pydantic_model("regulation")
        assert RegModel is not None
        reg = RegModel(name="GDPR")
        assert reg.name == "GDPR"

    def test_software_architecture_schema(self):
        path = SCHEMAS_DIR / "software-architecture.yaml"
        if not path.exists():
            pytest.skip("software-architecture.yaml not found")
        schema = load_domain_schema(path)
        assert schema.name == "software-architecture"
        names = schema.get_entity_type_names()
        assert "component" in names
        assert "apiendpoint" in names
        assert "technicaldebt" in names

    def test_software_architecture_models(self):
        path = SCHEMAS_DIR / "software-architecture.yaml"
        if not path.exists():
            pytest.skip("software-architecture.yaml not found")
        schema = load_domain_schema(path)
        CompModel = schema.get_pydantic_model("component")
        assert CompModel is not None
        comp = CompModel(name="auth-service")
        assert comp.name == "auth-service"

    def test_different_domains_have_different_scopes(self):
        pa_path = DOMAIN_YAML
        rc_path = SCHEMAS_DIR / "regulatory-compliance.yaml"
        if not pa_path.exists() or not rc_path.exists():
            pytest.skip("Required schema files not found")
        pa = load_domain_schema(pa_path)
        rc = load_domain_schema(rc_path)
        # Personal assistant uses personal/work; regulatory uses regulation/department
        assert pa.get_scopes() != rc.get_scopes()


# ── Tests: Framework Base Types Preserved ──────────────

class TestFrameworkBaseTypes:
    """Ensure Entity, FactMetadata, Relation base types still work."""

    def test_entity_base(self):
        from percos.models.ontology import Entity
        e = Entity(name="test")
        assert e.name == "test"
        assert e.id is not None
        assert e.metadata is not None
        assert e.tags == []

    def test_fact_metadata(self):
        from percos.models.ontology import FactMetadata
        fm = FactMetadata()
        assert fm.source == ""
        assert fm.created_at is not None
        assert fm.confidence is not None

    def test_relation_base(self):
        from uuid import uuid4
        from percos.models.ontology import Relation
        r = Relation(source_id=uuid4(), target_id=uuid4(), relation_type="test")
        assert r.relation_type == "test"
        assert r.weight == 1.0
