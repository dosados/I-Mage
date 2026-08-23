import threading

_ml_lock = threading.Lock()


def get_ml_lock() -> threading.Lock:
    return _ml_lock
