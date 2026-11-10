"""
The school's clock: the real time, taken from the internet, in the school's timezone.

The Pi's own clock can't be trusted (it has no battery to keep time while it's off, and its
timezone setting may be wrong), so the 4:15 p.m. cutoff, the 7:30 a.m. start and the clock on
the resting screen all use this instead of the Pi's clock.

How it works:
  - It asks internet time servers what time it really is (NTP, with the ntplib package). If the
    network blocks those (some school networks do), it reads the time from the "Date" stamp on
    an ordinary web page instead.
  - From that answer it keeps counting with time.monotonic(), a timer that changing the Pi's
    clock doesn't affect, and asks again every RESYNC_SECONDS.
  - The time is converted to TIMEZONE, so daylight saving switches over by itself.
  - Until the first answer arrives (or with no internet at all) it falls back to the Pi's clock,
    still in TIMEZONE, and says so in the terminal.

Use school_clock.now() anywhere the program needs the time of day.
"""
import email.utils
import threading
import time
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TIMEZONE = "America/New_York"   # the school's timezone (Eastern)
NTP_SERVERS = ("time.google.com", "time.cloudflare.com", "pool.ntp.org", "time.nist.gov")
WEB_SOURCES = ("https://www.google.com", "https://www.cloudflare.com", "https://www.microsoft.com")
TIMEOUT_SECONDS = 4
RESYNC_SECONDS = 30 * 60      # ask again this often, once it has an answer
RETRY_SECONDS = 60            # ...or this often while it can't get one

try:
    ZONE = ZoneInfo(TIMEZONE)
except Exception as error:   # Windows has no timezone list of its own: pip install tzdata
    raise SystemExit(f"Can't find the {TIMEZONE} timezone ({error}). Run: python -m pip install tzdata")
_base = None                  # (real UTC time in seconds, time.monotonic() at that moment)
_source = None                # where the last answer came from, for the terminal
_warned = False
_thread = None


def _from_ntp(server: str):
    import ntplib   # installed by installer.py; without it, the web fallback is used
    mono, wall = time.monotonic(), time.time()
    reply = ntplib.NTPClient().request(server, version=3, timeout=TIMEOUT_SECONDS)
    # reply.offset is how far the Pi's clock is off, measured at the moment of the request
    return wall + reply.offset, mono


def _from_web(url: str):
    """The "Date" header on any web response is the server's clock, to the second."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "hallway-filter-clock"})
    sent = time.monotonic()
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        stamp = response.headers.get("Date")
    got = time.monotonic()
    if not stamp:
        raise ValueError(f"no Date header from {url}")
    utc = email.utils.parsedate_to_datetime(stamp).timestamp()
    # The stamp is cut to the whole second and arrives half a round trip late
    return utc + 0.5 + (got - sent) / 2, got


def sync() -> bool:
    """Ask the internet what time it is. Returns True if something answered."""
    global _base, _source
    for name, ask, places in (("time server", _from_ntp, NTP_SERVERS), ("web", _from_web, WEB_SOURCES)):
        for place in places:
            try:
                utc, mono = ask(place)
            except Exception:   # unreachable, blocked, ntplib missing, odd reply: try the next one
                continue
            if not 1.6e9 < utc < 4.2e9:   # 2020 to 2103: anything else is a broken answer
                continue
            first = _base is None
            _base, _source = (utc, mono), f"{name} {place}"
            if first:
                print(f"School clock: {clock_text(now())} {TIMEZONE} (from {_source})")
            return True
    return False


def clock_text(t: datetime) -> str:
    """Tuesday 10 November, 4:15 p.m. (by hand: strftime's no-padding flags differ by OS)"""
    return f"{t:%A} {t.day} {t:%B}, {t.hour % 12 or 12}:{t.minute:02d} {'a.m.' if t.hour < 12 else 'p.m.'}"


def now() -> datetime:
    """The current time in the school's timezone, as a timezone-aware datetime."""
    global _warned
    if _base is not None:
        utc = _base[0] + (time.monotonic() - _base[1])
    else:
        if not _warned:
            _warned = True
            print("School clock: no internet time yet, using this computer's clock for now.")
        utc = time.time()
    return datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(ZONE)


def synced() -> bool:
    return _base is not None


def start() -> None:
    """Get the time in the background (so a missing internet connection doesn't hold up startup),
    then keep it fresh."""
    global _thread
    if _thread is not None:
        return

    def keep_fresh():
        while True:
            sync()
            time.sleep(RESYNC_SECONDS if _base is not None else RETRY_SECONDS)

    _thread = threading.Thread(target=keep_fresh, daemon=True)
    _thread.start()
