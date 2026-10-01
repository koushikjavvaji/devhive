import os
import time
from collections import defaultdict, deque

# every human message fans out to (1 + reaction rounds) calls per LLM teammate, so on a
# public deploy these are what stand between a spam script and the provider bill
MAX_MESSAGE_CHARS = int(os.environ.get("DEVHIVE_MAX_MESSAGE_CHARS", "8000"))
MESSAGES_PER_IP = int(os.environ.get("DEVHIVE_MESSAGES_PER_IP", "10"))
MESSAGES_WINDOW_SECONDS = int(os.environ.get("DEVHIVE_MESSAGES_WINDOW_SECONDS", "300"))
ROOMS_PER_IP_PER_HOUR = int(os.environ.get("DEVHIVE_ROOMS_PER_IP_PER_HOUR", "20"))
DAILY_MESSAGE_LIMIT = int(os.environ.get("DEVHIVE_DAILY_MESSAGE_LIMIT", "500"))

# how many proxies in front of the app append to X-Forwarded-For (Render: its load
# balancer). 0 = not behind a proxy, use the socket peer and ignore the header entirely.
TRUSTED_PROXY_HOPS = int(os.environ.get("DEVHIVE_TRUSTED_PROXY_HOPS", "1"))


class RateLimiter:
    """Sliding-window limiter: at most max_events per key within window_seconds. In-memory,
    so it resets on restart and isn't shared across processes — fine for a single small
    instance, which is what this deploys as."""

    def __init__(self, max_events, window_seconds, clock=time.monotonic):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.clock = clock
        self._events = defaultdict(deque)

    def allow(self, key="global"):
        now = self.clock()
        self._sweep(now)

        events = self._events[key]
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.max_events:
            return False
        events.append(now)
        return True

    def _sweep(self, now):
        # don't keep an entry around forever for every IP that ever showed up once
        if len(self._events) < 1000:
            return
        stale = [k for k, ev in self._events.items() if not ev or now - ev[-1] >= self.window_seconds]
        for key in stale:
            del self._events[key]


def client_ip(conn, trusted_hops=None):
    """Behind a proxy the socket peer is the proxy, so the client's IP comes from
    X-Forwarded-For — but only the entries our own proxies appended are trustworthy. The
    leftmost ones are whatever the client sent, so trusting those would let anyone dodge
    the per-IP limit by making up a new address each request."""
    hops = TRUSTED_PROXY_HOPS if trusted_hops is None else trusted_hops
    forwarded = conn.headers.get("x-forwarded-for")
    if hops > 0 and forwarded:
        entries = [e.strip() for e in forwarded.split(",") if e.strip()]
        if entries:
            return entries[-min(hops, len(entries))]
    return conn.client.host if conn.client else "unknown"
