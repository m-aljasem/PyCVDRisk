// Run seeded patients through the ASSIGN offline calculator, version 1.5.1 (March 2014), the developers' own code.
//
// Source: http://www.assign-score.com/downloads/assignscore-offline-version-1-5-1.zip (Wayback capture
// 20160404100837), unzipped to source/wayback/offline_1_5_1/. The calculator is a static HTML page whose score is
// computed by calculate() in includes/js/assign-offline.js with the constants in includes/js/betas.js. This script
// cuts out, verbatim, betas.js, the accMul/accAdd/accDiv helpers and the body of calculate() from the arthritis
// adjustment to the rounded score, and runs them in a Node vm with the form values as strings (as jQuery .val()
// returns them). Only the jQuery form reads and the result display are left out. The postcode lookup is replaced
// by setting simdScore directly (the page sets it from the SIMD 2012 postcode tables).
//
// Writes fixtures/offline_v151_reference.jsonl: inputs, the unrounded score 100*(1-c) and the displayed integer.
// Run from repo root: node reference/assign/run_offline_v151.js

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const js = path.join(__dirname, "source", "wayback", "offline_1_5_1", "includes", "js");
const app = fs.readFileSync(path.join(js, "assign-offline.js"), "utf8");
const slice = (from, to) => {
  const i = app.indexOf(from), j = app.indexOf(to, i);
  if (i < 0 || j < 0) throw new Error("marker not found: " + from);
  return app.slice(i, j);
};
const helpers = slice("function accMul", "var calculating");
const body = slice("if (arthritis > 0)", "var meansUsed");
const code = fs.readFileSync(path.join(js, "betas.js"), "utf8") + "\n" + helpers + "\n" +
  "function run(age, sex, chd, diabetes, arthritis, cpd, sbp, tc, hdl, simdScore) {\n" + body +
  "\nreturn {raw: accMul(100.0, accAdd(1.0, -c)), shown: score};\n}";
const ctx = {};
vm.createContext(ctx);
vm.runInContext(code, ctx);

// mulberry32, seed 151
let s = 151;
const rnd = () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const uni = (a, b, dp) => (a + rnd() * (b - a)).toFixed(dp);
const int = (a, b) => String(a + Math.floor(rnd() * (b - a + 1)));

const out = [];
for (let id = 0; id < 1000; id++) {
  const tc = uni(3.0, 9.0, 1);
  const smoker = rnd() < 0.35;
  const p = {id, sex: id % 2 ? "female" : "male", age: int(30, 74), total_chol: tc,
    hdl_chol: uni(0.6, Math.min(2.8, Number(tc) - 0.5), 2), sbp: int(95, 210), diabetes: rnd() < 0.15 ? "1" : "0",
    family_history: rnd() < 0.4 ? "1" : "0", cigarettes_per_day: smoker ? int(1, 40) : "0",
    rheumatoid_arthritis: rnd() < 0.2 ? "1" : "0", simd_score: Number(uni(0.5, 89, 2))};
  const r = ctx.run(p.age, p.sex, p.family_history, p.diabetes, p.rheumatoid_arthritis, p.cigarettes_per_day,
    p.sbp, p.total_chol, p.hdl_chol, p.simd_score);
  for (const k of ["age", "total_chol", "hdl_chol", "sbp", "diabetes", "family_history", "cigarettes_per_day",
    "rheumatoid_arthritis"]) p[k] = Number(p[k]);
  out.push(JSON.stringify({...p, raw: r.raw, shown: r.shown}));
}
fs.writeFileSync(path.join(__dirname, "fixtures", "offline_v151_reference.jsonl"), out.join("\n") + "\n");
console.log("wrote", out.length);
