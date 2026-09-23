from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from .config import load_settings
from .server import main as run_server
from .store import Store


def export_credentials(session_id: str | None, output: Path) -> None:
    settings = load_settings()
    store = Store(settings.database_path)
    session = store.get_session(session_id) if session_id else store.get_latest_session()
    if session is None:
        raise SystemExit("no session found to export")

    bundle = store.export_session(session.session_id)
    output.write_text(
        json.dumps(
            {
                "version": bundle.version,
                "exported_at": bundle.exported_at,
                "session": {
                    "session_id": bundle.session.session_id,
                    "access_token": bundle.session.access_token,
                    "refresh_token": bundle.session.refresh_token,
                    "token_type": bundle.session.token_type,
                    "expires_at": bundle.session.expires_at,
                    "created_at": bundle.session.created_at,
                },
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"exported session {bundle.session.session_id} to {output}")


def import_credentials(input_file: Path) -> None:
    settings = load_settings()
    store = Store(settings.database_path)
    bundle = json.loads(input_file.read_text(encoding="utf-8"))
    session = store.import_session(bundle)
    print(f"imported session {session.session_id} into {settings.database_path}")


def main() -> None:
    parser = ArgumentParser(prog="openai-oauth-proxy")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--host", type=str, help="Host to listen on")

    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export", help="Export stored credentials to JSON")
    export_parser.add_argument("--session-id", type=str, help="Session ID to export; defaults to the latest session")
    export_parser.add_argument("--output", type=Path, required=True, help="Output JSON file")

    import_parser = subparsers.add_parser("import", help="Import stored credentials from JSON")
    import_parser.add_argument("--input", type=Path, required=True, help="Input JSON file")

    args = parser.parse_args()

    if args.command == "export":
        export_credentials(args.session_id, args.output)
        return
    if args.command == "import":
        import_credentials(args.input)
        return

    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
