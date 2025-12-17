import time

_last_broker_success = 0
BROKER_STALE_AFTER = 30  # seconds


def mark_broker_success():
    global _last_broker_success
    _last_broker_success = time.time()


def broker_healthy():
    try:
        import fyers_api
    except ImportError:
        return False

    if _last_broker_success == 0:
        return False

    return (time.time() - _last_broker_success) < BROKER_STALE_AFTER
