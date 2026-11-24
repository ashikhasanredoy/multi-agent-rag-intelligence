#!/usr/bin/env python3
"""
Single-command launcher for the Multi-Agent RAG Intelligence Platform.

Usage:
    python run.py
    python run.py --port 8000
    python run.py --no-browser
    python run.py --host 127.0.0.1 --port 8080 --no-reload
"""

import sys
import time
import socket
import argparse
import threading
import webbrowser
from pathlib import Path

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def is_port_in_use(host: str, port: int) -> bool:
    """Check if the target port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def open_browser_delayed(url: str, delay_seconds: float = 1.5):
    """Open the web browser to the application URL after server initialization."""
    def _open():
        time.sleep(delay_seconds)
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def print_banner(host: str, port: int, model: str, provider: str):
    """Display an informative terminal banner with all service links."""
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    base_url = f"http://{display_host}:{port}"

    border = "=" * 70
    print(border)
    print("   🤖 Multi-Agent RAG Intelligence Platform")
    print("   LangGraph • Hybrid RAG • Cognitive Memory • Real-Time Tools")
    print(border)
    print(f"  ⚡ LLM Provider : {provider.upper()} ({model})")
    print(f"  🌐 Host & Port  : {host}:{port}")
    print(border)
    print("  📌 All Service Links:")
    print(f"  • Web Client UI   : {base_url}/")
    print(f"  • Swagger API Docs: {base_url}/docs")
    print(f"  • ReDoc Reference : {base_url}/redoc")
    print(f"  • System Health   : {base_url}/health")
    print(f"  • Documents API   : {base_url}/documents")
    print(f"  • Memory API      : {base_url}/memory")
    print(f"  • Chat Endpoint   : {base_url}/chat (POST)")
    print(border)
    print("  🚀 Starting Uvicorn server... (Press CTRL+C to stop)")
    print(border + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent RAG Intelligence Platform"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=True,
        help="Enable auto-reload on code changes (default: True)",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Do not automatically open the browser upon launch",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read active model and provider settings safely
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        model_name = settings.LLM_MODEL
        provider = getattr(settings, "LLM_PROVIDER", "openrouter")
    except Exception:
        model_name = "configured model"
        provider = "llm"

    # Check port availability
    if is_port_in_use(args.host, args.port):
        display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
        print(f"\n⚠️  Port {args.port} appears to already be in use!")
        print(f"   If an instance is already running, access it at: http://{display_host}:{args.port}/")
        print(f"   To run on a different port, use: python run.py --port {args.port + 1}\n")

    # Launch browser if requested
    if not args.no_browser:
        display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
        open_browser_delayed(f"http://{display_host}:{args.port}/")

    print_banner(args.host, args.port, model_name, provider)

    import uvicorn

    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\n🛑 Multi-Agent Platform shut down gracefully.")


if __name__ == "__main__":
    main()
