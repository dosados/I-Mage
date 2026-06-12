import argparse
import json
import sys

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def health(base_url: str) -> int:
    response = httpx.get(f"{base_url}/health", timeout=10.0)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


def search(base_url: str, query: str) -> int:
    response = httpx.post(
        f"{base_url}/search",
        json={"query": query},
        timeout=600.0,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="I-Mage API debug CLI")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="GET /health")

    search_parser = subparsers.add_parser("search", help="POST /search")
    search_parser.add_argument("query", help="Text search query")

    args = parser.parse_args()

    try:
        if args.command == "health":
            return health(args.base_url)
        if args.command == "search":
            return search(args.base_url, args.query)
    except httpx.HTTPError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
