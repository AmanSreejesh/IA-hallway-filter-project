"""
The school's clock: the real time in the school's timezone, without needing the internet.

The Pi's own clock can't be trusted, so the 4:15 p.m. cutoff, the 7:30 a.m. start and the clock
on the resting screen all use this instead. The Pi only needs internet once, during setup:

  1. Setup (with internet): run  python common/school_clock.py  (or just start main.py). It asks
     internet time servers what time it really is (NTP, with the ntplib package; if those are
     blocked, it reads the "Date" stamp on an ordinary web page) and saves the answer to
     common/clock.json.
  2. After that, no internet is needed. The saved answer is paired with the Pi's time-since-boot
     counter (time.monotonic()), which keeps counting correctly however wrong the Pi's clock is,
     including when main.py is closed and started again.
  3. If the Pi is turned off and on again, the time-since-boot counter starts over, so the saved
     difference between the real time and the Pi's clock is used instead. That's right as long
     as the Pi's clock kept counting while it was off (a Pi 5 with its clock battery does; a Pi
     without one restarts its clock from when it was shut down). If the time looks wrong after
     a power cut, connect the internet for a minute and it fixes itself.
  4. Whenever the internet happens to be there, it checks again and saves the new answer.

The time is converted to TIMEZONE, so daylight saving switches over by itself.
Use school_clock.now() anywhere the program needs the time of day.
"""
import email.utils
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TIMEZONE = "America/New_York"   # the school's timezone (Eastern)
NTP_SERVERS = ("time.google.com", "time.cloudflare.com", "pool.ntp.org", "time.nist.gov")
WEB_SOURCES = ("https://www.google.com", "https://www.cloudflare.com", "https://www.microsoft.com")
TIMEOUT_SECONDS = 4
RECHECK_SECONDS = 30 * 60     # how often to try the internet again (quietly; it's fine if it's not there)
FIRST_TRY_SECONDS = 60        # ...or this often while there's no saved time at all
SAVED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clock.json")
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"   # Linux: changes every time the computer starts

try:
    ZONE = ZoneInfo(TIMEZONE)
except Exception as error:   # Windows has no timezone list of its own: pip install tzdata
    raise SystemExit(f"Can't find the {TIMEZONE} timezone ({error}). Run: python -m pip install tzdata")

_base = None                  # (real UTC time in seconds, time.monotonic() at that moment)
_source = None                # where that came from, for the terminal and the staff panel
_loaded = False
_warned = False
_thread = None


# --- Asking the internet (setup, or whenever it happens to be connected) ---
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
    """Ask the internet what time it is and save the answer. Returns True if something answered.
    No internet is fine: it just returns False and the saved time carries on."""
    global _base, _source
    for name, ask, places in (("time server", _from_ntp, NTP_SERVERS), ("web", _from_web, WEB_SOURCES)):
        for place in places:
            try:
                utc, mono = ask(place)
            except Exception:   # offline, blocked, ntplib missing, odd reply: try the next one
                continue
            if not 1.6e9 < utc < 4.2e9:   # 2020 to 2103: anything else is a broken answer
                continue
            first = _source is None or not _source.startswith("internet")
            _base, _source = (utc, mono), f"internet ({name} {place})"
            _save(utc, mono)
            if first:
                print(f"School clock: {clock_text(now())} {TIMEZONE}, from the {_source[10:-1]}. Saved, "
                      "so it keeps the right time without the internet from now on.")
            return True
    return False


# --- Keeping it without the internet ---
def _boot_id():
    try:
        with open(BOOT_ID_PATH) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _save(utc: float, mono: float) -> None:
    data = {
        "utc": utc,                                          # the real time...
        "monotonic": mono,                                   # ...at this point of the time-since-boot counter
        "pi_clock": time.time() - (time.monotonic() - mono),   # ...and what the Pi's clock said then
        "boot_id": _boot_id(),
        "saved": clock_text(datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(ZONE)),
    }
    try:
        with open(SAVED_PATH + ".part", "w") as out:
            json.dump(data, out, indent=2)
        os.replace(SAVED_PATH + ".part", SAVED_PATH)
    except OSError as error:
        print(f"School clock: couldn't save the time ({error}); it will need the internet next start.")


def load_saved() -> bool:
    """Pick up the time saved at setup (or the last time the internet was connected)."""
    global _base, _source, _loaded
    _loaded = True
    try:
        with open(SAVED_PATH) as f:
            saved = json.load(f)
        utc, mono, pi_clock = (float(saved[k]) for k in ("utc", "monotonic", "pi_clock"))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    boot = _boot_id()
    if boot and saved.get("boot_id") == boot and time.monotonic() >= mono:
        # Same power-on as when it was saved: the time-since-boot counter is exact
        _base, _source = (utc, mono), f"saved {saved.get('saved', '')}"
    else:
        # The computer has restarted since: carry the saved difference over to the Pi's clock
        _base = (time.time() + (utc - pi_clock), time.monotonic())
        _source = f"saved {saved.get('saved', '')}, adjusted for a restart"
    print(f"School clock: {clock_text(now())} {TIMEZONE} (from the time {_source}). No internet needed.")
    return True


# --- Reading it ---
def clock_text(t: datetime) -> str:
    """Tuesday 10 November, 4:15 p.m. (by hand: strftime's no-padding flags differ by OS)"""
    return f"{t:%A} {t.day} {t:%B}, {t.hour % 12 or 12}:{t.minute:02d} {'a.m.' if t.hour < 12 else 'p.m.'}"


def now() -> datetime:
    """The current time in the school's timezone, as a timezone-aware datetime."""
    global _warned
    if _base is None and not _loaded:
        load_saved()
    if _base is not None:
        utc = _base[0] + (time.monotonic() - _base[1])
    else:
        if not _warned:
            _warned = True
            print("School clock: no saved time yet, so this computer's clock is used. Connect the "
                  "internet once (during setup) and it saves the real time.")
        utc = time.time()
    return datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(ZONE)


def source() -> str:
    """Where the time is coming from, for the staff panel."""
    return _source or "this computer's clock (never connected to the internet)"


def start() -> None:
    """Load the saved time straight away, then check the internet in the background whenever it's
    there (so a missing connection never holds anything up)."""
    global _thread
    if _thread is not None:
        return
    if not _loaded:
        load_saved()

    def recheck():
        while True:
            sync()
            time.sleep(RECHECK_SECONDS if _base is not None else FIRST_TRY_SECONDS)

    _thread = threading.Thread(target=recheck, daemon=True)
    _thread.start()


if __name__ == "__main__":
    # Setup: python common/school_clock.py (with the internet connected)
    if sync():
        print(f"Done. Saved to {SAVED_PATH}. The Pi can go offline now.")
    else:
        print("Couldn't reach the internet. Connect it and run this again.")
        raise SystemExit(1)
