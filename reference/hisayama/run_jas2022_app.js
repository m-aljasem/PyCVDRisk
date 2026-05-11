// Run the Japan Atherosclerosis Society's official 2022 risk app (web version of the "これりすくん" tool for
// physicians, https://www.j-athero.org/general/gl2022app/workers/, framed by https://www.j-athero.org/jp/general/ge_tool2/).
// The app computes in the browser: source/jas2022_app/mainQuestions.js (calsSuitaScore2 + createResult2) is loaded
// unmodified in a vm sandbox with a minimal jQuery stand-in that returns the form values; the query string it builds
// for result.html (rate, rateText, risk, riskText, rateMin) is what the result page displays.
// Patients: every point cell (4 ages x 2 sexes x 3 HDL x 4 LDL x 5 SBP x 2 glucose x 2 smoking = 1,920) at band
// boundary values, plus 2,000 seeded patients with raw values, half entering TC + TG (Friedewald route) instead of LDL-C.
// Run from repo root: node reference/hisayama/run_jas2022_app.js
const fs = require("fs"), path = require("path"), vm = require("vm");
const SRC = path.join(__dirname, "source/jas2022_app/mainQuestions.js");
const OUT = path.join(__dirname, "fixtures/jas2022_app_reference.json");

function run(form) {
  const alerts = [];
  const $ = (sel) => {
    if (typeof sel === "function") return;  // document-ready handlers (UI only)
    const name = /\[name=(\w+)\]/.exec(sel)[1];
    return { val: () => (form[name] === undefined ? "" : String(form[name])) };
  };
  const ctx = { $, showAlert: (m) => alerts.push(m), encodeURIComponent, parseInt, console };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(SRC, "utf8"), ctx);
  const q = ctx.calsSuitaScore2();
  if (typeof q !== "string") return { error: alerts.join(" ") };
  const p = Object.fromEntries(q.slice(1).split("&").map((kv) => kv.split("=").map(decodeURIComponent)));
  return { rate: +p.rate, rateText: p.rateText, risk: +p.risk, riskText: p.riskText, rateMin: +p.rateMin, ldl_used: +p.ldl };
}

let s = 2022;  // mulberry32
const rnd = () => { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const int = (a, b) => a + Math.floor(rnd() * (b - a + 1));

const pts = [];
for (const old of [45, 55, 65, 75]) for (const sex of [0, 7]) for (const hdl of [39, 40, 60])
  for (const ldl of [119, 120, 140, 160]) for (const sbp of [119, 120, 130, 140, 160])
    for (const dm of [0, 1]) for (const smoking of [0, 2]) pts.push({ old, sex, hdl, ldl, systolicBloodPressure: sbp, diabeteMellitus: dm, smoking });
for (let i = 0; i < 2000; i++) {
  const f = { old: [45, 55, 65, 75][int(0, 3)], sex: [0, 7][int(0, 1)], hdl: int(25, 100), systolicBloodPressure: int(95, 200),
    diabeteMellitus: int(0, 1), smoking: [0, 2][int(0, 1)] };
  if (i % 2) { f.tc = int(f.hdl + 80, 320); f.tg = int(40, 400); } else f.ldl = int(60, 230);
  pts.push(f);
}
const rows = pts.map((f, id) => ({ id, ...f, ...run(f) }));
fs.writeFileSync(OUT, JSON.stringify({ source: "https://www.j-athero.org/general/gl2022app/workers/mainQuestions.js",
  sha256: require("crypto").createHash("sha256").update(fs.readFileSync(SRC)).digest("hex"), n: rows.length, rows }, null, 0) + "\n");
console.log(rows.length, "patients;", rows.filter((r) => r.error).length, "errors");
