// Illustrations for the resting screen: a cartoon person wearing each filter.
// Pure drawing (SVG strings). app.js builds the cards and runs the "filter goes on" sweep.
// Each picture uses viewBox 0 0 200 220 and the same face layout as the others:
//   head centre (100, 102), eyes at (80, 102) and (120, 102), nose tip (100, 118).
"use strict";

(function () {
  // Four different visitors, so the screen looks like the hallway it sits in
  const PEOPLE = {
    emoji: { skin: "#F1C27D", shade: "#D49A5A", hair: "#3B2A20", style: "short", shirt: "#3B4A63" },
    sharingan: { skin: "#8D5524", shade: "#6E3F17", hair: "#17120E", style: "curly", shirt: "#4A3B5C" },
    helmet: { skin: "#E0AC69", shade: "#BF884A", hair: "#16110D", style: "bun", shirt: "#2D3A4F" },
    clown: { skin: "#FFDBAC", shade: "#E8B98A", hair: "#B45309", style: "long", shirt: "#3D4F3A" },
    other: { skin: "#C68642", shade: "#A36A2E", hair: "#2B1D14", style: "short", shirt: "#3A3F4B" },
  };

  const HAIR = {
    short: (c) => `<path d="M50 98C45 56 76 38 102 40c30 2 54 22 48 58-6-22-22-32-46-32-22 0-44 6-54 32Z" fill="${c}"/>`,
    curly: (c) => [[58, 74, 17], [76, 56, 19], [100, 49, 20], [124, 56, 19], [142, 74, 17], [50, 94, 11], [150, 94, 11], [88, 50, 14], [112, 50, 14]]
      .map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`).join(""),
    bun: (c) => `<circle cx="100" cy="40" r="15" fill="${c}"/><path d="M50 98C46 58 74 42 100 42s54 16 50 56c-8-20-26-30-50-30S58 78 50 98Z" fill="${c}"/>`,
    long: (c) => `<path d="M50 100C48 60 74 42 100 42s52 18 50 58c-10-20-30-34-50-36-14 10-32 20-50 36Z" fill="${c}"/>`,
  };
  const HAIR_BACK = {
    long: (c) => `<path d="M47 96C45 58 70 36 100 36s55 22 53 60l5 74c-18 9-98 9-116 0Z" fill="${c}"/>`,
  };

  function person(p, id) {
    return `
    <g class="person">
      <path d="M26 224c4-40 36-58 74-58s70 18 74 58Z" fill="${p.shirt}"/>
      <path d="M26 224c4-40 36-58 74-58s70 18 74 58Z" fill="url(#${id}-cloth)"/>
      <path d="M86 138h28v28c0 7-28 7-28 0Z" fill="${p.shade}"/>
      ${(HAIR_BACK[p.style] || (() => ""))(p.hair)}
      <ellipse cx="51" cy="106" rx="8" ry="12" fill="${p.shade}"/>
      <ellipse cx="149" cy="106" rx="8" ry="12" fill="${p.shade}"/>
      <ellipse cx="100" cy="102" rx="50" ry="58" fill="${p.skin}"/>
      <ellipse cx="100" cy="102" rx="50" ry="58" fill="url(#${id}-skin)"/>
      ${HAIR[p.style](p.hair)}
      <circle cx="70" cy="122" r="8" fill="#FF8787" opacity=".28"/>
      <circle cx="130" cy="122" r="8" fill="#FF8787" opacity=".28"/>
      <g class="blink">
        <ellipse cx="80" cy="102" rx="12" ry="9" fill="#fff"/>
        <ellipse cx="120" cy="102" rx="12" ry="9" fill="#fff"/>
        <circle cx="80" cy="102" r="6.5" fill="#4A3426"/><circle cx="120" cy="102" r="6.5" fill="#4A3426"/>
        <circle cx="80" cy="102" r="3" fill="#120c08"/><circle cx="120" cy="102" r="3" fill="#120c08"/>
        <circle cx="82.5" cy="99.5" r="1.8" fill="#fff"/><circle cx="122.5" cy="99.5" r="1.8" fill="#fff"/>
      </g>
      <path d="M68 86q12-7 24-1M108 85q12-6 24 1" stroke="${p.hair}" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="M100 106q-6 11 0 14q3 1 6-1" stroke="${p.shade}" stroke-width="3" fill="none" stroke-linecap="round"/>
      <path d="M86 136q14 11 28 0" stroke="#7A2E2E" stroke-width="3.5" fill="none" stroke-linecap="round"/>
    </g>`;
  }

  const heart = (x, y, s) =>
    `<path d="M${x} ${y + 7 * s}C${x - 15 * s} ${y - 3 * s} ${x - 9 * s} ${y - 16 * s} ${x} ${y - 7 * s}C${x + 9 * s} ${y - 16 * s} ${x + 15 * s} ${y - 3 * s} ${x} ${y + 7 * s}Z" fill="#E03131"/>`;

  // The emoji filter gives each person a random emoji and swaps it every few seconds,
  // so the picture cycles through a few faces too (app.js moves .expr.on along).
  function emoji(id) {
    const ink = "#5B3A10";
    return `
    <defs>
      <radialGradient id="${id}-emo" cx="40%" cy="34%" r="72%">
        <stop offset="0" stop-color="#FFE38A"/><stop offset=".55" stop-color="#FCC419"/><stop offset="1" stop-color="#EE9A0E"/>
      </radialGradient>
    </defs>
    <g class="emo">
      <circle cx="100" cy="102" r="69" fill="url(#${id}-emo)"/>
      <ellipse cx="74" cy="60" rx="15" ry="8" fill="#fff" opacity=".45" transform="rotate(-30 74 60)"/>
      <g class="expr on">
        <ellipse cx="78" cy="88" rx="7" ry="11" fill="${ink}"/><ellipse cx="122" cy="88" rx="7" ry="11" fill="${ink}"/>
        <path d="M64 110H136Q132 146 100 146T64 110Z" fill="${ink}"/>
        <path d="M69 112H131Q130 121 127 124H73Q70 121 69 112Z" fill="#fff"/>
      </g>
      <g class="expr">
        ${heart(78, 88, 1.15)}${heart(122, 88, 1.15)}
        <path d="M72 118Q100 140 128 118" stroke="${ink}" stroke-width="6" fill="none" stroke-linecap="round"/>
      </g>
      <g class="expr">
        <path d="M54 80H95V86Q95 104 76 104T54 86ZM105 80H146V86Q146 104 127 104T105 86Z" fill="#1B1B1B"/>
        <path d="M95 84H105M54 82L38 78M146 82L162 78" stroke="#1B1B1B" stroke-width="4" stroke-linecap="round"/>
        <path d="M62 86L72 82M113 86L123 82" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".5"/>
        <path d="M80 126Q102 136 122 120" stroke="${ink}" stroke-width="6" fill="none" stroke-linecap="round"/>
      </g>
      <g class="expr">
        <ellipse cx="78" cy="88" rx="7" ry="11" fill="${ink}"/>
        <path d="M112 90Q122 82 132 90" stroke="${ink}" stroke-width="6" fill="none" stroke-linecap="round"/>
        <path d="M68 112Q100 148 132 112Z" fill="${ink}"/>
        <path d="M88 126Q100 142 112 126Q100 120 88 126Z" fill="#F06595"/>
      </g>
    </g>`;
  }

  function tomoeIris(cx, cy, r, cls) {
    const t = r / 6.5;   // drawn for an iris of radius 6.5, scaled to r
    const comma = (a) =>
      `<path transform="rotate(${a})" fill="#120808" d="M${-1.3 * t} ${-3.9 * t}A${1.3 * t} ${1.3 * t} 0 1 0 ${1.3 * t} ${-3.9 * t}` +
      `Q${1.1 * t} ${-5.9 * t} ${-1.8 * t} ${-5.9 * t}Q${-0.7 * t} ${-5.2 * t} ${-1.3 * t} ${-3.9 * t}Z"/>`;
    return `
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="#E03131"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#1a0606" stroke-width="${0.9 * t}"/>
      <circle cx="${cx}" cy="${cy}" r="${3.9 * t}" fill="none" stroke="#1a0606" stroke-width="${0.45 * t}" opacity=".8"/>
      <g class="${cls}" transform="translate(${cx} ${cy})"><g class="spin">${[0, 120, 240].map(comma).join("")}</g></g>
      <circle cx="${cx}" cy="${cy}" r="${1.7 * t}" fill="#120808"/>`;
  }

  // Sharingan is all about the eye, so its picture is one big eye instead of a person.
  // Drawn around (100, 110) and moved down a little to sit in the middle of the frame.
  // The iris is clipped to the eyelids, like the real filter.
  const EYE = "M14 110Q100 34 186 110Q100 178 14 110Z";
  function eyeBase(id) {
    const lashes = [[40, 91, 31, 79], [66, 78, 60, 63], [88, 73, 85, 57], [112, 73, 115, 57], [134, 78, 140, 63], [160, 91, 169, 79]]   // on the lid line
      .map(([x1, y1, x2, y2]) => `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="#1D1B18" stroke-width="4" stroke-linecap="round"/>`).join("");
    return `
    <defs>
      <clipPath id="${id}-lid"><path d="${EYE}"/></clipPath>
      <radialGradient id="${id}-white" cx="50%" cy="45%" r="60%"><stop offset=".6" stop-color="#FFFDF8"/><stop offset="1" stop-color="#E9DCCB"/></radialGradient>
    </defs>
    <g transform="translate(0 16)">
    <path d="M32 46Q100 8 170 42" stroke="#2A1C14" stroke-width="12" fill="none" stroke-linecap="round"/>
    <g class="blink">
      <path d="${EYE}" fill="url(#${id}-white)"/>
      <g clip-path="url(#${id}-lid)">
        <circle cx="100" cy="110" r="42" fill="#5A3A22"/>
        <circle cx="100" cy="110" r="42" fill="none" stroke="#2E1C10" stroke-width="4"/>
        <circle cx="100" cy="110" r="17" fill="#120C08"/>
        <circle cx="115" cy="95" r="8" fill="#fff" opacity=".9"/>
      </g>
      <path d="M14 110Q100 34 186 110" stroke="#1D1B18" stroke-width="6" fill="none" stroke-linecap="round"/>
      <path d="M22 113Q100 172 178 113" stroke="#1D1B18" stroke-width="2.5" fill="none" stroke-linecap="round" opacity=".55"/>
      ${lashes}
    </g></g>`;
  }

  function sharingan(id) {
    return `
    <defs><clipPath id="${id}-lid2"><path d="${EYE}"/></clipPath></defs>
    <g transform="translate(0 16)"><g class="blink"><g clip-path="url(#${id}-lid2)">
      ${tomoeIris(100, 110, 42, "tomoe")}
      <circle cx="115" cy="95" r="8" fill="#fff" opacity=".8"/>
    </g>
    <path d="M14 110Q100 34 186 110" stroke="#1D1B18" stroke-width="6" fill="none" stroke-linecap="round"/></g></g>`;
  }

  // Helmet: a 3D space helmet with a see-through visor (the face stays visible)
  function helmet(id) {
    return `
    <defs>
      <radialGradient id="${id}-shell" cx="35%" cy="28%" r="80%">
        <stop offset="0" stop-color="#ffffff"/><stop offset=".6" stop-color="#E9ECEF"/><stop offset="1" stop-color="#ADB5BD"/>
      </radialGradient>
      <clipPath id="${id}-visor"><ellipse cx="100" cy="106" rx="54" ry="48"/></clipPath>
    </defs>
    <path d="M36 180q64 26 128 0l8 22q-72 30-144 0Z" fill="#CED4DA" stroke="#868E96" stroke-width="2"/>
    <path d="M40 186q60 22 120 0" stroke="#1C7ED6" stroke-width="4" fill="none"/>
    <path fill-rule="evenodd" fill="url(#${id}-shell)" stroke="#868E96" stroke-width="2"
      d="M100 22a78 78 0 1 1 0 156a78 78 0 1 1 0-156ZM100 58a54 48 0 1 0 0.01 0Z"/>
    <path d="M40 66q60-60 120 0" stroke="#1C7ED6" stroke-width="6" fill="none" stroke-linecap="round"/>
    <rect x="14" y="88" width="14" height="34" rx="6" fill="#DEE2E6" stroke="#868E96" stroke-width="2"/>
    <rect x="172" y="88" width="14" height="34" rx="6" fill="#DEE2E6" stroke="#868E96" stroke-width="2"/>
    <circle class="beacon" cx="21" cy="96" r="3.5" fill="#FA5252"/>
    <circle class="beacon b2" cx="179" cy="96" r="3.5" fill="#40C057"/>
    <ellipse cx="100" cy="106" rx="54" ry="48" fill="#4DABF7" opacity=".16"/>
    <g clip-path="url(#${id}-visor)">
      <path d="M58 72q20-22 52-20" stroke="#fff" stroke-width="7" fill="none" stroke-linecap="round" opacity=".6"/>
      <rect class="glint" x="20" y="40" width="22" height="140" fill="#fff" opacity=".35" transform="skewX(-20)"/>
    </g>
    <ellipse cx="100" cy="106" rx="54" ry="48" fill="none" stroke="#343A40" stroke-width="5"/>`;
  }

  function clown(id) {
    return `
    <defs>
      <radialGradient id="${id}-nose" cx="36%" cy="32%" r="70%">
        <stop offset="0" stop-color="#FF8787"/><stop offset=".5" stop-color="#F03E3E"/><stop offset="1" stop-color="#A51111"/>
      </radialGradient>
    </defs>
    <g class="boing">
      <circle cx="100" cy="116" r="16" fill="url(#${id}-nose)"/>
      <ellipse cx="94" cy="109" rx="5.5" ry="3.8" fill="#fff" opacity=".75" transform="rotate(-25 94 109)"/>
    </g>`;
  }

  const FX = { emoji, sharingan, helmet, clown };

  // One card's picture: the person, then the filter layer (the part that sweeps on)
  function picture(key, id, color) {
    const p = PEOPLE[key] || PEOPLE.other;
    const fx = FX[key] ? FX[key](id) : "";
    return `<svg viewBox="0 0 200 220" aria-hidden="true">
      <defs>
        <radialGradient id="${id}-skin" cx="40%" cy="35%" r="70%"><stop offset=".6" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".16"/></radialGradient>
        <linearGradient id="${id}-cloth" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".12"/><stop offset="1" stop-color="#000" stop-opacity=".25"/></linearGradient>
        <clipPath id="${id}-wipe"><rect class="wipe" x="-40" y="-40" width="280" height="300"/></clipPath>
      </defs>
      ${key === "sharingan" ? eyeBase(id) : person(p, id)}
      <g class="fx" clip-path="url(#${id}-wipe)">${fx}</g>
      <g class="scan" opacity="0">
        <rect x="-40" y="-9" width="280" height="18" fill="${color}" opacity=".35"/>
        <rect x="-40" y="-1.5" width="280" height="3" fill="#fff"/>
      </g>
    </svg>`;
  }

  window.RestArt = { picture, known: (key) => key in FX };
})();
