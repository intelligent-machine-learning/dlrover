from datetime import datetime, timedelta


def has_expired(timestamp: float, time_period: int) -> bool:
    dt = datetime.fromtimestamp(timestamp)
    expired_dt = dt + timedelta(seconds=time_period)
    return expired_dt < datetime.now()
