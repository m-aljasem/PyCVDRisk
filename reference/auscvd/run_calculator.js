// Run the Aus CVD Risk Calculator's own equations and write their results.
// Run from repo root: node reference/auscvd/run_calculator.js
//
// source/calculator-078e1c450ffb448c.js is the calculator page's bundle from www.cvdcheck.org.au/calculator
// (unmodified). Its four equation functions (MenGeneralEquation, WomenGeneralEquation,
// MenDiabetesEquation, WomenDiabetesEquation) and utility_getRecalibrationFactor are cut from it byte for byte
// and run in a vm context. Each is fed the values the page's handleSubmit builds: seifa = 6 - SEIFA quintile,
// smoker / exSmoker from the smoking choice, HdlRatio, cvd_medicine (the ticked medicines) and BPLM / LLM / OATM,
// ACR_3_to_30 = 3 <= uACR <= 30 and ACR_gt_30 = uACR > 30, eGFR (100 when "90 or more" is ticked).
// Each patient is written as shown (Math.round of max(1, risk) * factor, a whole percent) and with Math.round
// replaced by the identity.
const fs = require("fs");
const vm = require("vm");
const dir = "reference/auscvd";
const src = fs.readFileSync(`${dir}/source/calculator-078e1c450ffb448c.js`, "utf8");
const cut = (from, to) => { const i = src.indexOf(from), j = src.indexOf(to, i); if (i < 0 || j < 0) throw from; return src.slice(i, j); };
const ctx = {};
vm.createContext(ctx);
vm.runInContext(cut("var utility_getRecalibrationFactor=", ";let M=(0,o.k)"), ctx);
vm.runInContext("var " + cut("MenGeneralEquation=e=>", ";return(0,f.useEffect)"), ctx);
const M = vm.runInContext("Math", ctx), round = M.round;

let seed = 2023;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const pick = (a) => a[Math.floor(rand() * a.length)];
const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const MEDS = ["Blood pressure-lowering medicines", "Lipid-modifying medicines", "Antithrombotic medicines"];
const UACR = [0.1, 1, 2.9, 3, 3.1, 10, 30, 30.1, 100, 1000];

const rows = [];
for (let i = 0; i < 6000; i++) {
  const diab = i >= 3000 ? 1 : rand() < 0.2 ? 1 : 0;
  const p = {
    sex: pick(["0", "1"]), age: i % 50 === 0 ? pick([44, 45, 54, 55, 59, 60, 64, 65, 69, 70]) : int(30, 79),
    quintile: int(1, 5), smoking: pick(["0", "1", "2"]), sbp: int(75, 220), tc_hdl: +(1.5 + 8 * rand()).toFixed(1),
    af: rand() < 0.1 ? 1 : 0, bp: rand() < 0.4 ? 1 : 0, lipid: rand() < 0.3 ? 1 : 0, antithrombotic: rand() < 0.2 ? 1 : 0,
    diabetes: diab,
  };
  const equation = diab && i >= 3000 ? "diabetes" : "general";
  if (equation === "diabetes") {
    Object.assign(p, { years: int(0, 40), bmi: +(17 + 33 * rand()).toFixed(2), egfr: rand() < 0.3 ? 100 : int(10, 130),
      hba1c: int(30, 130), uacr: pick(UACR), insulin: rand() < 0.2 ? 1 : 0 });
  }
  const e = { sex: p.sex, age: p.age, seifa: 6 - p.quintile, smoker: p.smoking === "2" ? 1 : 0,
    exSmoker: p.smoking === "1" ? 1 : 0, sbp: p.sbp, HdlRatio: p.tc_hdl, af: p.af, diabetes: p.diabetes,
    cvd_medicine: MEDS.filter((_, k) => [p.bp, p.lipid, p.antithrombotic][k]), BPLM: p.bp, LLM: p.lipid, OATM: p.antithrombotic };
  if (equation === "diabetes") {
    Object.assign(e, { year: p.years, bmi: p.bmi, eGFR: p.egfr, HbA1c: p.hba1c, insulin: p.insulin,
      ACR_3_to_30: p.uacr >= 3 && p.uacr <= 30 ? 1 : 0, ACR_gt_30: p.uacr > 30 ? 1 : 0 });
  }
  const fn = (p.sex === "1" ? "Men" : "Women") + (equation === "diabetes" ? "DiabetesEquation" : "GeneralEquation");
  M.round = round;
  const shown = ctx[fn](e);
  M.round = (x) => x;
  const unrounded = ctx[fn](e);
  M.round = round;
  rows.push({ ...p, equation, factor: ctx.utility_getRecalibrationFactor(p.sex === "0" ? "female" : "male", p.age),
    shown, unrounded: unrounded.toPrecision(17) });
}
const cols = ["equation", "sex", "age", "quintile", "smoking", "sbp", "tc_hdl", "af", "bp", "lipid", "antithrombotic", "diabetes",
  "years", "bmi", "egfr", "hba1c", "uacr", "insulin", "factor", "shown", "unrounded"];
fs.writeFileSync(`${dir}/fixtures/calculator_reference.csv`,
  [cols.join(","), ...rows.map((r) => cols.map((c) => r[c] ?? "").join(","))].join("\n") + "\n");
console.log(`wrote ${rows.length} rows`);
