from threading import Lock

from pipeline.orchestrator import ThinkSymOrchestrator


_store_lock = Lock()
_orchestrators: dict[str, ThinkSymOrchestrator] = {}


def get_orchestrator(session_id: str, device: str = "cuda") -> ThinkSymOrchestrator:
    key = session_id.strip() or "default"
    with _store_lock:
        current = _orchestrators.get(key)
        if current is None:
            current = ThinkSymOrchestrator(session_id=key, device=device)
            _orchestrators[key] = current
        return current


def reset_session(session_id: str) -> bool:
    key = session_id.strip() or "default"
    with _store_lock:
        current = _orchestrators.get(key)
        if current is None:
            return False
        current.reset_session()
        return True


def clear_all() -> None:
    with _store_lock:
        _orchestrators.clear()
