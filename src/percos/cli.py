"""CLI entry point for the PCOS server and utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from percos.config import get_settings


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the PCOS API server."""
    settings = get_settings()
    if args.schema:
        import os
        os.environ["DOMAIN_SCHEMA_PATH"] = args.schema
    uvicorn.run(
        "percos.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.value.lower(),
    )


def cmd_init(args: argparse.Namespace) -> None:
    """Initialise a new PCOS data directory and optional domain schema."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "chroma").mkdir(parents=True, exist_ok=True)

    schema_dest = Path(args.schema_out) if args.schema_out else data_dir / "domain.yaml"
    if not schema_dest.exists():
        # Copy the bundled personal-assistant schema as starter
        bundled = Path(__file__).resolve().parent.parent.parent / "schemas" / "personal-assistant.yaml"
        if not bundled.exists():
            bundled = Path(__file__).resolve().parent.parent.parent / "domain.yaml"
        if bundled.exists():
            schema_dest.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Created domain schema: {schema_dest}")
        else:
            print(f"Warning: no bundled schema found. Create {schema_dest} manually.")
    else:
        print(f"Schema already exists: {schema_dest}")

    env_file = data_dir / ".env"
    if not env_file.exists():
        env_file.write_text(
            "# PCOS environment variables\n"
            f"DOMAIN_SCHEMA_PATH={schema_dest}\n"
            "OPENAI_API_KEY=\n"
            f"DATABASE_URL=sqlite+aiosqlite:///{data_dir / 'percos.db'}\n",
            encoding="utf-8",
        )
        print(f"Created .env: {env_file}")
    else:
        print(f".env already exists: {env_file}")

    print("Initialisation complete.")


def cmd_validate_schema(args: argparse.Namespace) -> None:
    """Validate a domain schema YAML file."""
    from percos.schema import parse_schema_dict

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"Error: file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    schema, errors = parse_schema_dict(raw)
    if errors:
        print(f"Schema validation failed ({len(errors)} error(s)):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # Show summary
    print(f"Schema OK: {schema.name} v{schema.version}")
    print(f"  Entity types: {len(schema.entity_types)}")
    print(f"  Relations:    {len(schema.relation_types)}")
    print(f"  Scopes:       {', '.join(schema.scopes)}")
    model_map = schema.get_model_map()
    print(f"  Models generated: {len(model_map)}")
    if args.json:
        print(json.dumps(schema.to_dict(), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="percos",
        description="PCOS – Personal Cognitive OS",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--schema", default=None, help="Path to domain schema YAML")
    p_serve.add_argument("--reload", action="store_true", default=True)
    p_serve.set_defaults(func=cmd_serve)

    # init
    p_init = sub.add_parser("init", help="Initialise data directory & schema")
    p_init.add_argument("--data-dir", default="./data")
    p_init.add_argument("--schema-out", default=None, help="Where to write starter schema")
    p_init.set_defaults(func=cmd_init)

    # validate-schema
    p_val = sub.add_parser("validate-schema", help="Validate a domain schema YAML")
    p_val.add_argument("schema", help="Path to YAML schema file")
    p_val.add_argument("--json", action="store_true", help="Print parsed schema as JSON")
    p_val.set_defaults(func=cmd_validate_schema)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        # Default: start server (backward compatible)
        args_ns = argparse.Namespace(host=None, port=None, schema=None, reload=True)
        cmd_serve(args_ns)


if __name__ == "__main__":
    main()
