// The filter wheel. Talks to common/dashboard_server.py: asks what's on about once a second,
// and tells it when someone picks a filter.
"use strict";

const $ = (id) => document.getElementById(id);
const wheel = $("wheel");

// ---------- how each filter looks on the wheel, and who made it ----------
const LOOKS = {
  emoji: { label: "Emoji", by: "Viraj", color: "#E8590C", icon: '<circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r="1.3" fill="#fff" stroke="none"/><circle cx="15" cy="10" r="1.3" fill="#fff" stroke="none"/><path d="M8 14.5c1 1.6 2.4 2.4 4 2.4s3-.8 4-2.4"/>' },
  sharingan: { label: "Sharingan", by: "Aman", color: "#D6336C", icon: '<path d="M2.5 12C5 7.5 8.3 5.5 12 5.5s7 2 9.5 6.5C19 16.5 15.7 18.5 12 18.5S5 16.5 2.5 12Z"/><circle cx="12" cy="12" r="3.6"/><circle cx="12" cy="12" r="1.2" fill="#fff" stroke="none"/>' },
  helmet: { label: "Astronaut Helmet", by: "Ishan", color: "#1C7ED6", icon: '<circle cx="12" cy="11" r="8"/><rect x="7" y="8" width="10" height="6" rx="3"/><path d="M7 21h10"/>' },
  clown: { label: "Clown Nose", by: "Aryan", color: "#2F9E44", icon: '<circle cx="12" cy="13" r="6.5"/><path d="M9.3 11a3 3 0 0 1 2.7-1.9"/>' },
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
  if (demo) rollDemo(now);
  else if (!drag) step(angle, dt);
  step(hubPop, dt);
  step(flash, dt);
  draw();
  const still = [scale, angle, hubPop, flash].every((s) => Math.abs(s.t - s.x) < 0.002 && Math.abs(s.v) < 0.01);
  if (still && !drag && !demo) {   // everything has settled: stop drawing until the next touch (saves the Pi)
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
  $("name").innerHTML = f.by ? `${f.label} <span class="by">by ${f.by}</span>` : f.name.replace(/[<>&"]/g, "");
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
  filters = names.map((n, i) => ({ name: n, label: n, ...look(n, i) }));
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
  // Their filter has just started (the privacy note is done): show them the wheel rolling to it
  if (demoPending != null && state.in_use && !resting) {
    runDemo(demoPending);
    demoPending = null;
  }
  if (state.resting) demoPending = null;
  // The filter changed on its own (taking turns, or a key press): spin the wheel to it
  if (!drag && !demo && performance.now() - lastPick > 2000 && sliceAt(angle.t) !== state.active) {
    spinTo(state.active);
    if (!open) showName();
  }
  showMessage();
  setClock(state);
  updateRest();
  $("off").classList.toggle("on", state.off);
  $("firm").classList.toggle("on", state.firm && !state.off);
  $("debug").textContent = `Debug view: ${state.debug ? "on" : "off"}`;
  $("staffInfo").textContent = `${state.tracker}, ${state.fps} frames a second. ${state.zone_note || ""}`;
  $("staffClock").textContent = state.clock_source ? `Clock: ${state.clock_source}` : "";
  updatePrivacy();
  if (sel < 0) draw();
}

// The little label beside the small wheel, while someone has a filter on
function updateTip() {
  $("wheelTip").classList.toggle("on", !!state && !!state.in_use && !resting && !open && !demo && !state.off);
}

function showMessage() {
  updateTip();
  const m = $("message");
  let text = "";
  if (isLost) text = "Reconnecting…";
  else if (state.off || state.firm) text = "";
  else if ((open || demo) && !resting) text = "Drag the wheel to switch filters";
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

// ---------- the wheel shows itself off once ----------
// Right after the privacy note, the wheel opens to full size, rolls round a couple of times and
// lands on the filter they picked on the resting screen (or a random one), then shrinks back.
// That way everyone sees there's a wheel and that it turns.
const DEMO_TURNS = 2;
const DEMO_MS = 3200;
let demo = null;          // { from, to, start }
let demoPending = null;   // filter to land on once their filter starts
let demoTimer = 0;

function runDemo(i) {
  if (!filters.length) return;
  const base = -mod(i, filters.length) * span;
  const to = angle.x - mod(angle.x - base, 360) - 360 * DEMO_TURNS;   // lands exactly on slice i
  angle.v = 0;
  demo = { from: angle.x, to, start: performance.now() + 450 };       // let it open first
  lastPick = performance.now() + DEMO_MS + 2500;                      // don't let updates turn it meanwhile
  setOpen(true);
  clearTimeout(idleTimer);
  clearTimeout(demoTimer);
  showMessage();
  kick();
}

function rollDemo(now) {
  const p = Math.max(0, Math.min(1, (now - demo.start) / DEMO_MS));
  const e = 1 - Math.pow(1 - p, 4);   // quick at first, then slows right down onto the slice
  angle.x = angle.t = demo.from + (demo.to - demo.from) * e;
  angle.v = 0;
  if (p >= 1) {
    demo = null;
    hubPop.v += 10;
    demoTimer = setTimeout(() => { setOpen(false); showMessage(); }, 1700);
  }
}

function stopDemo() {
  if (!demo && !demoTimer) return;
  demo = null;
  clearTimeout(demoTimer);
  demoTimer = 0;
  angle.t = Math.round(angle.x / span) * span;
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
  if (state) showMessage();
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
  stopDemo();
  angle.t = angle.x;
  angle.v = 0;
  // Started on the small wheel: it grows as you drag, so follow the finger sideways (its centre
  // moves while it grows). On the open wheel, follow the finger round the ring.
  drag = { prev: pointerAngle(e), t: performance.now(), v: 0, x: e.clientX, y: e.clientY, lastX: e.clientX,
           real: false, sideways: !open };
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
    open = true;                // dragging the small wheel opens it, so you can see what you're turning
    clearTimeout(idleTimer);
    scale.t = 1.05;             // lifts a little while it spins
    showMessage();
  }
  let d;
  if (drag.sideways) {
    // Moving the finger right turns the top of the wheel right, like dragging a dial
    const ring = Math.max(40, (size * 92 / 260) * scale.x);
    d = ((e.clientX - drag.lastX) / ring) * (180 / Math.PI);
    drag.lastX = e.clientX;
  } else {
    if (Math.hypot(e.clientX - cx, e.clientY - cy) < radius * 0.15) return;   // too close to the middle to tell direction
    const g = pointerAngle(e);
    d = g - drag.prev;
    if (d > 180) d -= 360;
    if (d < -180) d += 360;
    drag.prev = g;
  }
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

  if (resting) {
    privacyAway = true;   // the resting screen has its own privacy line
    lastBusy = now - PRIVACY_IDLE_BACK;
  } else if (intro || state.in_use) {
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
  if (resting && (k === " " || k === "enter")) { startFromRest(null); return; }
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


// ---------- resting screen ----------
// Up whenever main.py says nobody's using the display (state.resting). The filters take turns in
// the big frame: each one opens out of the middle, starts as a plain face, and a line sweeps
// across to put the filter on. A tap anywhere tells main.py to start (tapping a filter starts with
// that one); main.py brings this screen back once nobody has been in view for 5 seconds.
const REST = {
  emoji: { label: "Emoji", by: "Viraj", color: "#E9A23B", tint: "#F1DDB6", what: "Your whole head turns into an emoji, and you get a new one every few seconds." },
  sharingan: { label: "Sharingan", by: "Aman", color: "#B8433A", tint: "#EBCFC6", what: "From the anime Naruto." },
  helmet: { label: "Astronaut Helmet", by: "Ishan", color: "#3A6EA5", tint: "#D3DDE8", what: "Your own space helmet. The visor's clear, so it's still you inside." },
  clown: { label: "Clown Nose", by: "Aryan", color: "#4C8C5A", tint: "#D3E1CF", what: "A big red nose that stays put, however much you wiggle." },
};
const REST_SPARE = [["#7A5BA6", "#DDD3E6"], ["#2F7F7F", "#CFE1DE"], ["#A6623A", "#EBD8C8"]];
const REST_TAP_GRACE = 2500;      // ms to trust a tap before main.py's next update confirms it
let resting = false;
let restNames = "";
let restItems = [];
let restLit = -1;
let restTapAt = -1e9;
let restTimer = 0;
let restSweep = 0;
let emojiTimer = 0;

function restKey(name) {
  return Object.keys(REST).find((k) => name.toLowerCase().includes(k)) || "other";
}

function restTurnMs() {
  const v = parseFloat(getComputedStyle($("rest")).getPropertyValue("--turn"));
  return (Number.isFinite(v) && v > 0 ? v : 5) * 1000;
}

function buildRest(names) {
  restItems = names.map((name, i) => {
    const key = restKey(name);
    const spare = REST_SPARE[i % REST_SPARE.length];
    const r = REST[key] || { color: spare[0], tint: spare[1], what: "Tap to try it on." };
    const safe = name.replace(/[<>&"]/g, "");
    return { name: safe, label: safe, key, ...r };
  });
  $("restStage").innerHTML = restItems.map((f, i) =>
    `<div class="slide" style="--c:${f.color}">${RestArt.picture(f.key, "st" + i, f.color)}<span class="slide-line"></span></div>`).join("");
  $("restRail").innerHTML = restItems.map((f, i) =>
    `<div class="thumb" role="listitem" data-index="${i}" aria-label="${f.label}${f.by ? " by " + f.by : ""}: tap to try">
      <div class="thumb-frame" style="--c:${f.color}">${RestArt.picture(f.key, "th" + i, f.color)}</div>
      <p class="thumb-name">${f.label}</p>${f.by ? `<p class="thumb-by">by ${f.by}</p>` : ""}<div class="thumb-bar"></div>
    </div>`).join("");
  restNames = names.join("|");
  restLit = -1;
}

// The next filter opens out of the middle of the big frame
function showRest(i) {
  if (!restItems.length) return;
  const prev = restLit;
  restLit = mod(i, restItems.length);
  const f = restItems[restLit];
  const slides = [...$("restStage").children];
  slides.forEach((el, j) => {
    el.classList.remove("now", "under");
    if (j === prev && j !== restLit) el.classList.add("under");
  });
  const slide = slides[restLit];
  void slide.offsetWidth;   // start the opening from nothing, even if it was just showing
  slide.classList.add("now");
  $("rest").style.setProperty("--tint", f.tint);
  [...$("restRail").children].forEach((t, j) => {
    t.classList.remove("now");
    if (j === restLit) { void t.offsetWidth; t.classList.add("now"); }   // restarts its timer line
  });
  const cap = $("restCaption");
  cap.classList.add("swap");
  setTimeout(() => {
    $("restName").innerHTML = f.by ? `${f.label} <span class="rest-by">by ${f.by}</span>` : f.label;
    $("restWhat").textContent = f.what;
    cap.classList.remove("swap");
  }, prev < 0 ? 0 : 260);
  sweep(slide);
}

// Plain face first, then a line glides across and the filter goes on behind it
function sweep(slide) {
  cancelAnimationFrame(restSweep);
  const wipe = slide.querySelector(".wipe");
  const line = slide.querySelector(".slide-line");
  const scan = slide.querySelector(".scan");
  if (scan) scan.setAttribute("opacity", "0");
  if (!wipe || !line) return;
  const begin = performance.now() + 1100, ms = 1400;
  wipe.setAttribute("x", "240");
  line.style.left = "100%";
  line.style.opacity = "0";
  const tick = (now) => {
    const p = Math.max(0, Math.min(1, (now - begin) / ms));
    const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
    const x = 200 * (1 - e);
    wipe.setAttribute("x", String(p >= 1 ? -40 : x));
    line.style.left = `${(x / 200) * 100}%`;
    line.style.opacity = p > 0 && p < 1 ? "1" : "0";
    if (p < 1) restSweep = requestAnimationFrame(tick);
  };
  restSweep = requestAnimationFrame(tick);
}

// The emoji in the big frame swaps faces, like the real filter swapping emojis
function nextEmoji() {
  document.querySelectorAll(".rest-stage .emo").forEach((g) => {
    const faces = [...g.querySelectorAll(".expr")];
    const k = faces.findIndex((f) => f.classList.contains("on"));
    faces.forEach((f, j) => f.classList.toggle("on", j === (k + 1) % faces.length));
  });
}

function updateRest() {
  const want = !!state.resting && !state.off && performance.now() - restTapAt > REST_TAP_GRACE;
  if (state.filters.join("|") !== restNames) {
    buildRest(state.filters);
    if (resting) showRest(0);
  }
  const rest = $("rest");
  rest.classList.toggle("noticed", want && state.people > 0);   // someone's in view: the tap ring hurries
  if (want === resting) return;
  resting = want;
  document.body.classList.toggle("resting", resting);
  clearInterval(restTimer);
  clearInterval(emojiTimer);
  if (resting) {
    rest.classList.remove("leaving");
    rest.classList.add("on");
    if (open) setOpen(false);
    showRest(state.active ?? 0);
    restTimer = setInterval(() => showRest(restLit + 1), restTurnMs());
    emojiTimer = setInterval(nextEmoji, 1600);
  } else {
    rest.classList.remove("on");
    cancelAnimationFrame(restSweep);
  }
}

function startFromRest(index) {
  if (!resting) return;
  restTapAt = performance.now();
  // Tapped somewhere other than a filter: pick one for them at random
  const picked = index != null ? index : Math.floor(Math.random() * Math.max(1, restItems.length));
  send("start", picked);
  lastPick = performance.now();
  if (state) state.active = picked;
  demoPending = picked;             // the wheel rolls to it once the privacy note is done
  const rest = $("rest");
  rest.classList.add("leaving");
  resting = false;
  clearInterval(restTimer);
  clearInterval(emojiTimer);
  cancelAnimationFrame(restSweep);
  setTimeout(() => {
    document.body.classList.remove("resting");
    rest.classList.remove("on", "leaving");
    updatePrivacy();
  }, 450);
}

// The clock in the corner shows school time, the same clock the 4:15 cutoff uses: main.py sends
// the real (internet) time and the school's timezone, because this computer's clock can't be
// trusted. Blank until the first update arrives, rather than showing the wrong time.
let clockOffset = null;   // ms to add to this computer's clock to get the real time
let clockZone;
function tickClock() {
  if (clockOffset == null) return;
  const now = new Date(Date.now() + clockOffset);
  try {
    $("restTime").textContent = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", timeZone: clockZone });
    $("restDate").textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric", timeZone: clockZone });
  } catch {   // an unknown timezone name: show it in this computer's zone rather than nothing
    clockZone = undefined;
  }
}
function setClock(s) {
  if (typeof s.clock_ms !== "number") return;
  clockOffset = s.clock_ms - Date.now();
  clockZone = s.timezone || undefined;
  tickClock();
}
setInterval(tickClock, 5000);

// The whole screen is the button. A filter (in the big frame or down the side) starts with that one
$("rest").addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  goFullscreen();
  const thumb = e.target.closest(".thumb");
  if (thumb) return startFromRest(Number(thumb.dataset.index));
  if (e.target.closest(".rest-stage") && restLit >= 0) return startFromRest(restLit);
  startFromRest(null);
});
