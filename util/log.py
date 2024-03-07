from datetime import datetime


def log(*msg):
    print(f"[{datetime.now()}] -", " ".join(map(str, msg)))
