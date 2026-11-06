// The filter wheel. Talks to common/dashboard_server.py: asks what's on about once a second,
// and tells it when someone picks a filter.
"use strict";

const $ = (id) => document.getElementById(id);
const wheel = $("wheel");

// ---------- how each filter looks on the wheel ----------
const LOOKS = {
  emoji: { color: "#E8590C", icon: '<circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r="1.3" fill="#fff" stroke="none"/><circle cx="15" cy="10" r="1.3" fill="#fff" stroke="none"/><path d="M8 14.5c1 1.6 2.4 2.4 4 2.4s3-.8 4-2.4"/>' },
  sharingan: { color: "#D6336C", icon: '<path d="M2.5 12C5 7.5 8.3 5.5 12 5.5s7 2 9.5 6.5C19 16.5 15.7 18.5 12 18.5S5 16.5 2.5 12Z"/><circle cx="12" cy="12" r="3.6"/><circle cx="12" cy="12" r="1.2" fill="#fff" stroke="none"/>' },
  helmet: { color: "#1C7ED6", icon: '<circle cx="12" cy="11" r="8"/><rect x="7" y="8" width="10" height="6" rx="3"/><path d="M7 21h10"/>' },
  clown: { color: "#2F9E44", icon: '<circle cx="12" cy="13" r="6.5"/><path d="M9.3 11a3 3 0 0 1 2.7-1.9"/>' },
};
const SPARE_COLORS = ["#7048E8", "#0B7285", "#C2255C", "#5C940D"];

function look(name, i) {
  const key = Object.keys(LOOKS).find((k) => name.toLowerCase().includes(k));
  if (key) return LOOKS[key];
  const letter = (name.trim()[0] || "?").toUpperCase().replace(/[<>&"]/g, "");
  return {
    color: SPARE_COLORS[i % SPARE_COLORS.length],
    icon: `<text x="12" y="16.5" text-anchor="middle" font-size="13" font-weight="700" fill="#fff" stroke="none" font-family="system-ui, sans-serif">${letter}</text>`,
  };
}
const iconSvg = (inner) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;

// ---------- springs: every movement eases with a little bounce ----------
const spring = (x, k, c) => ({ x, v: 0, t: x, k, c });
const scale = spring(0.34, 260, 17);   // small <-> open
const angle = spring(0, 190, 20);     // the wheel's turn, in degrees
const hubPop = spring(1, 420, 14);    // the middle pops when a new filter lands
const flash = spring(0, 300, 26);     // the name, shown for a moment when the wheel is small
const SMALL = 0.34;

let filters = [];     // [{name, color, icon}]
let span = 360;       // degrees per slice
let sel = -1;         // slice under the pointer
let open = false;
let drag = null;
let running = false;
let last = 0;
let size = wheel.offsetWidth;
let lastPick = 0;     // when someone last touched the wheel (the program's answer can lag behind)
let state = null;
let idleTimer = 0;
let flashTimer = 0;

function kick() {
  if (!running) {
    running = true;
    last = performance.now();
    requestAnimationFrame(frame);
  }
}

function step(s, dt) {
  s.v += (s.k * (s.t - s.x) - s.c * s.v) * dt;
  s.x += s.v * dt;
}

function frame(now) {
  const dt = Math.min(0.032, (now - last) / 1000);
  last = now;
  step(scale, dt);
  if (!drag) step(angle, dt);
  step(hubPop, dt);
  step(flash, dt);
  draw();
  const still = [scale, angle, hubPop, flash].every((s) => Math.abs(s.t - s.x) < 0.002 && Math.abs(s.v) < 0.01);
  if (still && !drag) {   // everything has settled: stop drawing until the next touch (saves the Pi)
    [scale, angle, hubPop, flash].forEach((s) => { s.x = s.t; s.v = 0; });
    draw();
    running = false;
    return;
  }
  requestAnimationFrame(frame);
}

const mod = (a, n) => ((a % n) + n) % n;
const sliceAt = (a) => mod(Math.round(-a / span), filters.length || 1);

function draw() {
  if (!filters.length) return;
  const a = angle.x;
  wheel.style.transform = `scale(${scale.x})`;
  $("segLayer").style.transform = `rotate(${a}deg)`;
  $("ticksLayer").style.transform = `rotate(${-a * 0.4}deg)`;
  $("dashLayer").style.transform = `rotate(${a * 1.6}deg)`;

  const r = (size * 92) / 260;
  const showIcons = Math.max(0, Math.min(1, (scale.x - 0.35) / 0.25));
  [...$("icons").children].forEach((el, i) => {
    const d = ((i * span + a) * Math.PI) / 180;
    el.style.transform = `translate(${r * Math.sin(d)}px, ${-r * Math.cos(d)}px) scale(${i === sel ? 1.15 : 0.9})`;
    el.style.opacity = showIcons;
  });

  const k = sliceAt(a);
  if (k !== sel) choose(k);
  $("hub").style.transform = `scale(${hubPop.x})`;

  const name = $("name");
  const openness = Math.max(0, Math.min(1, (scale.x - 0.5) / 0.3));
  const shown = Math.max(openness, flash.x);
  const top = parseFloat(getComputedStyle(wheel).bottom) + size * scale.x * 1.1 + 12;   // clears the pointer above the ring
  name.style.opacity = shown;
  name.style.transform = `translate(-50%, ${-top}px) scale(${0.8 + 0.2 * shown})`;
}

// A new slice is under the pointer
function choose(k) {
  sel = k;
  const f = filters[k];
  $("hub").innerHTML = iconSvg(f.icon);
  $("name").textContent = f.name;
  $("pointer").setAttribute("fill", f.color);
  [...$("segs").children].forEach((p, i) => {
    p.style.strokeWidth = i === k ? 46 : 38;
    p.style.strokeOpacity = i === k ? 1 : 0.5;
  });
  [...$("icons").children].forEach((el, i) => el.setAttribute("aria-checked", i === k));
  hubPop.v += 6;
  kick();
}

// ---------- building the wheel ----------
function polar(r, deg) {
  const d = (deg * Math.PI) / 180;
  return [r * Math.sin(d), -r * Math.cos(d)];
}
function arc(r, a0, a1) {
  const [x0, y0] = polar(r, a0), [x1, y1] = polar(r, a1);
  return `M${x0} ${y0}A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1} ${y1}`;
}

(function buildTicks() {
  let t = "";
  for (let d = 0; d < 360; d += 6) {
    const long = d % 30 === 0;
    const [x0, y0] = polar(122, d), [x1, y1] = polar(long ? 112 : 117, d);
    t += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#fff" stroke-opacity="${long ? 0.8 : 0.35}" stroke-width="1.5"/>`;
  }
  $("ticks").innerHTML = t + '<circle r="127" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="1"/>';
})();

function buildWheel(names) {
  filters = names.map((n, i) => ({ name: n, ...look(n, i) }));
  span = 360 / filters.length;
  const gap = filters.length > 1 ? 4 : 0.01;
  $("segs").innerHTML = filters.map((f, i) =>
    `<path d="${arc(92, i * span - span / 2 + gap, i * span + span / 2 - gap)}" fill="none" stroke="${f.color}" stroke-width="38" stroke-opacity=".5"/>`).join("");
  $("icons").innerHTML = filters.map((f, i) =>
    `<div class="icon" role="radio" aria-label="${f.name.replace(/"/g, "")}">${iconSvg(f.icon)}</div>`).join("");
  sel = -1;
  draw();
}

// ---------- talking to main.py ----------
function send(action, value) {
  return fetch("api/do", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, value }),
  }).catch(() => {});
}

async function poll() {
  try {
    const res = await fetch("api/state", { cache: "no-store" });
    const next = await res.json();
    if (next && next.filters) {
      state = next;
      render();
    }
    lost(false);
  } catch {
    lost(true);
  }
}
setInterval(poll, 1000);
poll();

let isLost = false;
function lost(v) { isLost = v; if (state) showMessage(); }

const feed = $("feed");
feed.addEventListener("error", () => {
  lost(true);
  setTimeout(() => { feed.src = "stream?" + Date.now(); }, 2000);
});

function render() {
  if (filters.map((f) => f.name).join("|") !== state.filters.join("|")) buildWheel(state.filters);
  // The filter changed on its own (taking turns, or a key press): spin the wheel to it
  if (!drag && performance.now() - lastPick > 2000 && sliceAt(angle.t) !== state.active) {
    spinTo(state.active);
    if (!open) showName();
  }
  showMessage();
  $("off").classList.toggle("on", state.off);
  $("firm").classList.toggle("on", state.firm && !state.off);
  $("debug").textContent = `Debug view: ${state.debug ? "on" : "off"}`;
  $("staffInfo").textContent = `${state.tracker}, ${state.fps} frames a second. ${state.zone_note || ""}`;
  updatePrivacy();
  if (sel < 0) draw();
}

function showMessage() {
  const m = $("message");
  let text = "";
  if (isLost) text = "Reconnecting…";
  else if (state.off || state.firm) text = "";
  else if (state.gentle) text = "1 minute has passed";
  else if (state.hint) text = "Come closer";
  if (text) m.textContent = text;
  m.classList.toggle("on", !!text);
}

function showName() {
  flash.t = 1;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { flash.t = 0; kick(); }, 1400);
  kick();
}

// Turn the shortest way round to slice i
function spinTo(i) {
  let target = -i * span;
  while (target - angle.x > 180) target -= 360;
  while (angle.x - target > 180) target += 360;
  angle.t = target;
  kick();
}

function picked() {
  lastPick = performance.now();
  const i = sliceAt(angle.t);
  if (state && (state.auto || state.active !== i)) send("select", i);
  if (!open) showName();
}

// ---------- touch ----------
function setOpen(v) {
  open = v;
  scale.t = v ? 1 : SMALL;
  clearTimeout(idleTimer);
  if (v) idleTimer = setTimeout(() => setOpen(false), 5000);
  kick();
}

function center() {
  const r = wheel.getBoundingClientRect();
  return [r.left + r.width / 2, r.top + r.height / 2, r.width / 2];
}
function pointerAngle(e) {
  const [cx, cy] = center();
  return (Math.atan2(e.clientX - cx, -(e.clientY - cy)) * 180) / Math.PI;
}

wheel.addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  goFullscreen();
  try { wheel.setPointerCapture(e.pointerId); } catch {}
  angle.t = angle.x;
  angle.v = 0;
  drag = { prev: pointerAngle(e), t: performance.now(), v: 0, x: e.clientX, y: e.clientY, real: false };
  hubPop.t = 0.9;
  scale.t = (open ? 1 : SMALL) * 0.94;   // squish under the finger
  clearTimeout(idleTimer);
  kick();
});

wheel.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const [cx, cy, radius] = center();
  if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 6 && !drag.real) {
    drag.real = true;
    scale.t = (open ? 1 : SMALL) * 1.05;   // lifts a little while it spins
  }
  if (Math.hypot(e.clientX - cx, e.clientY - cy) < radius * 0.15) return;   // too close to the middle to tell direction
  const g = pointerAngle(e);
  let d = g - drag.prev;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  drag.prev = g;
  const now = performance.now();
  drag.v = drag.v * 0.6 + ((d / Math.max(1, now - drag.t)) * 1000) * 0.4;
  drag.t = now;
  angle.x += d;
  kick();
});

function release(e, cancelled) {
  if (!drag) return;
  const d = drag;
  drag = null;
  hubPop.t = 1;
  if (d.real || cancelled) {
    // Let go: keep the spin going a little, then click into the nearest slice
    const v = performance.now() - d.t > 80 ? 0 : d.v;
    angle.v = v;
    angle.t = Math.round((angle.x + v * 0.12) / span) * span;
    scale.t = open ? 1 : SMALL;
    if (open) setOpen(true);
    picked();
  } else if (!open) {
    setOpen(true);
  } else {
    const [cx, cy] = center();
    const dist = Math.hypot(e.clientX - cx, e.clientY - cy) / scale.x;
    if (dist < size * 0.17) {
      setOpen(false);   // tapped the middle
    } else {
      spinTo(mod(Math.round((pointerAngle(e) - angle.x) / span), filters.length));
      picked();
      setTimeout(() => setOpen(false), 650);
    }
  }
  kick();
}
wheel.addEventListener("pointerup", (e) => release(e, false));
wheel.addEventListener("pointercancel", (e) => release(e, true));

// Tapping the picture closes the wheel. The first tap also makes the page full screen.
function goFullscreen() {
  if (!document.fullscreenElement && $("staff").hidden && !isKiosk()) {
    document.documentElement.requestFullscreen?.().catch(() => {});
  }
}
document.addEventListener("pointerdown", () => {
  goFullscreen();
  if (open) setOpen(false);
});
function isKiosk() { return window.innerHeight >= screen.height - 2 && window.innerWidth >= screen.width - 2; }

window.addEventListener("resize", () => { size = wheel.offsetWidth; draw(); });
document.addEventListener("contextmenu", (e) => e.preventDefault());


// ---------- privacy: corner card, and the 3-second intro before a filter starts ----------
// The corner card is up whenever nobody is using a filter. When someone walks up, main.py waits
// DWELL_SECONDS before starting their filter (so people hurrying past never get one), and the page
// shows the privacy note in the middle with a countdown meanwhile. The corner card comes back once
// nobody has been in view for a moment, or nobody has used a filter for a while.
const PRIVACY_BACK_AFTER = 2;       // seconds with nobody in view (measured by main.py)
const PRIVACY_IDLE_BACK = 5000;     // ms with nobody using a filter
let privacyAway = false;
let lastBusy = 0;
let introStart = null;
let introMs = 3000;

function updatePrivacy() {
  if (!state) return;
  const now = performance.now();
  const intro = state.intro != null && !isLost && !state.off;
  introMs = (state.dwell || 3) * 1000;
  if (intro) {
    if (introStart == null) introStart = now - state.intro * introMs;
  } else {
    introStart = null;
  }
  $("intro").classList.toggle("on", intro && performance.now() - introStart < introMs);
  showIntro();

  if (intro || state.in_use) {
    privacyAway = true;
    lastBusy = now;
  } else if (privacyAway && (isLost || (state.empty_for ?? 0) >= PRIVACY_BACK_AFTER || now - lastBusy >= PRIVACY_IDLE_BACK)) {
    privacyAway = false;
  }
  $("privacy").classList.toggle("away", privacyAway);
}

function showIntro() {
  if (introStart == null) return;
  const done = Math.min(1, (performance.now() - introStart) / introMs);
  $("introFill").style.strokeDashoffset = String(125.66 * (1 - done));
  $("introNum").textContent = String(Math.max(1, Math.ceil((introMs / 1000) * (1 - done))));
  if (done >= 1) $("intro").classList.remove("on");   // countdown over: don't wait for the next update
}
setInterval(showIntro, 50);

// ---------- staff: hold the top-left corner for 2 seconds ----------
let holdTimer = 0;
$("corner").addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  holdTimer = setTimeout(() => { $("staff").hidden = false; $("staff").classList.add("on"); }, 2000);
});
["pointerup", "pointerleave", "pointercancel"].forEach((ev) =>
  $("corner").addEventListener(ev, () => clearTimeout(holdTimer)));

$("staff").addEventListener("pointerdown", (e) => e.stopPropagation());
$("staff").addEventListener("click", (e) => {
  const b = e.target.closest("[data-staff]");
  if (!b) return;
  const what = b.dataset.staff;
  if (what === "close") { $("staff").hidden = true; $("staff").classList.remove("on"); }
  else if (what === "fullscreen") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.().catch(() => {});
  } else if (what === "calibrate") {
    send("calibrate");
    b.textContent = "Calibrated";
    setTimeout(() => { b.textContent = "Calibrate (look straight at the screen)"; }, 2000);
  } else {
    send(what).then(poll);
  }
});

// ---------- keyboard, for testing at a desk ----------
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || !filters.length) return;
  const k = e.key.toLowerCase();
  if (k === "arrowright" || k === "arrowleft") {
    angle.t = Math.round(angle.t / span) * span + (k === "arrowright" ? -span : span);
    kick();
    picked();
  } else if (k === "n") send("next");
  else if (k === "d") send("debug").then(poll);
  else if (k === "c") send("calibrate");
  else if (k === "f") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.().catch(() => {});
  } else if (k === "escape") setOpen(false);
});

// Hide the mouse pointer when nobody is using it
let idleCursor = 0;
document.addEventListener("pointermove", () => {
  document.body.classList.remove("idle");
  clearTimeout(idleCursor);
  idleCursor = setTimeout(() => document.body.classList.add("idle"), 3000);
});
