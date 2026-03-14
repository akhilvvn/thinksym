from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
API_PORT = "8002"


def load_env_file() -> dict[str, str]:
    env_path = ROOT / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value
            loaded[key] = value

    return loaded


def start_backend(env_overrides: dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            API_PORT,
            "--reload",
        ],
        cwd=str(ROOT),
        env=env,
    )


def start_frontend(env_overrides: dict[str, str] | None = None) -> subprocess.Popen:
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env["NEXT_PUBLIC_API_BASE_URL"] = f"http://127.0.0.1:{API_PORT}"
    return subprocess.Popen([npm_cmd, "run", "dev"], cwd=str(FRONTEND_DIR), env=env)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    if not FRONTEND_DIR.exists():
        print("frontend directory not found", file=sys.stderr)
        return 1

    env_overrides = load_env_file()

    backend = start_backend(env_overrides)
    frontend = start_frontend(env_overrides)

    print("ThinkSym enterprise stack started")
    print(f"API: http://127.0.0.1:{API_PORT}")
    print("Console: http://localhost:3000")
    print("Press Ctrl+C to stop both processes")

    try:
        while True:
            backend_code = backend.poll()
            frontend_code = frontend.poll()

            if backend_code is not None:
                stop_process(frontend)
                return backend_code

            if frontend_code is not None:
                stop_process(backend)
                return frontend_code

            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_process(frontend)
        stop_process(backend)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())