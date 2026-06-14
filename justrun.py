#!/usr/bin/env python3
"""
justrun.py — One-command setup and launch for the PDF Intelligence Platform.

Run this file once after cloning and it will:
  1. Check prerequisites (Python, pip, Ollama)
  2. Prompt you for any required API keys
  3. Install all Python dependencies
  4. Pull the Llama model if not already present
  5. Launch the app in your browser at http://localhost:8501
"""

import getpass
import os
import shutil
import subprocess
import sys


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          PDF Intelligence Platform — Quick Start            ║
║    Progressive Learning · Judge AI · RAG · 100% Local       ║
╚══════════════════════════════════════════════════════════════╝
"""

OLLAMA_MODEL = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest"


def step(label: str) -> None:
    print(f"\n── {label} {'─' * max(0, 52 - len(label))}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def die(msg: str) -> None:
    print(f"\n  ✗  {msg}")
    sys.exit(1)


# ── 0. Banner ─────────────────────────────────────────────────────────────────

def show_banner() -> None:
    print(BANNER)
    print("  This script sets up and launches the PDF Intelligence Platform.")
    print("  All AI runs locally — no cloud LLM costs after setup.\n")


# ── 1. Prerequisites ──────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    step("Checking prerequisites")

    if sys.version_info < (3, 9):
        die(f"Python 3.9+ required. You have {sys.version.split()[0]}.")
    ok(f"Python {sys.version.split()[0]}")

    pip = shutil.which("pip3") or shutil.which("pip")
    if not pip:
        die("pip not found. Install it with your Python distribution.")
    ok("pip found")

    if not shutil.which("ollama"):
        print("\n  Ollama is not installed.")
        print("  Install it from: https://ollama.ai")
        print("  Then re-run this script.\n")
        die("Ollama is required for local AI inference.")
    ok("Ollama found")

    if not os.path.exists("app.py"):
        die("Run this script from the project root directory (where app.py lives).")
    ok("Running from correct directory")


# ── 2. API keys ───────────────────────────────────────────────────────────────

def _read_existing_env() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def setup_api_keys() -> None:
    step("API Keys Setup")
    print("  Keys are saved to your local .env file (never committed to git).\n")

    existing = _read_existing_env()

    # LlamaCloud API key (optional)
    print("  LlamaCloud API Key")
    print("  → Enables high-quality PDF parsing (tables, equations, captions).")
    print("  → Get a free key at: https://cloud.llamaindex.ai")
    print("  → Press Enter to skip — PyPDF fallback will be used instead.\n")

    existing_llama = existing.get("LLAMA_CLOUD_API_KEY", "")
    if existing_llama:
        keep = input(
            f"  Found existing key ending in …{existing_llama[-4:]}. Keep it? [Y/n]: "
        ).strip().lower()
        llama_key = existing_llama if keep != "n" else getpass.getpass("  New LlamaCloud API Key: ").strip()
    else:
        llama_key = getpass.getpass("  LlamaCloud API Key (Enter to skip): ").strip()

    env_lines = [f"LLAMA_CLOUD_API_KEY={llama_key}\n"]

    with open(".env", "w") as f:
        f.writelines(env_lines)

    if llama_key:
        ok("LlamaCloud API key saved to .env")
    else:
        info("Skipped — will use PyPDF fallback parser")


# ── 3. Python dependencies ─────────────────────────────────────────────────────

def install_requirements() -> None:
    step("Installing Python dependencies")

    pip = shutil.which("pip3") or shutil.which("pip")
    result = subprocess.run(
        [pip, "install", "-r", "requirements.txt", "-q", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:])
        die("Dependency installation failed. See error above.")
    ok("All dependencies installed")


# ── 4. Ollama model ───────────────────────────────────────────────────────────

def ensure_ollama_model() -> None:
    step("Checking Ollama model")

    # Check that Ollama server is responding
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            die("Ollama is installed but not running. Start it with: ollama serve")
    except subprocess.TimeoutExpired:
        die("Ollama is not responding. Start it with: ollama serve")

    if OLLAMA_MODEL in result.stdout:
        ok(f"Model already present: {OLLAMA_MODEL}")
        return

    print(f"\n  Model not found locally: {OLLAMA_MODEL}")
    print("  Pulling now — this may take a few minutes on first run…\n")

    pull = subprocess.run(["ollama", "pull", OLLAMA_MODEL], timeout=600)
    if pull.returncode != 0:
        warn(f"Model pull failed. Try manually: ollama pull {OLLAMA_MODEL}")
    else:
        ok(f"Model ready: {OLLAMA_MODEL}")


# ── 5. Launch ─────────────────────────────────────────────────────────────────

def launch_app() -> None:
    step("Launching app")

    print("  Opening browser at http://localhost:8501")
    print("  Press Ctrl+C to stop the server.\n")

    # Prefer the streamlit binary; fall back to `python3 -m streamlit` when the
    # binary isn't on PATH (common with pip3 user installs on macOS).
    streamlit_bin = shutil.which("streamlit")
    if streamlit_bin:
        cmd = [streamlit_bin, "run", "app.py"]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]

    subprocess.run(cmd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    show_banner()
    check_prerequisites()
    setup_api_keys()
    install_requirements()
    ensure_ollama_model()
    launch_app()


if __name__ == "__main__":
    main()
