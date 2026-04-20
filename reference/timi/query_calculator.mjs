// Drive the TIMI Study Group's own calculator app (https://timibiostat.shinyapps.io/calculators/, linked from
// https://timi.org/timi-calculators/), tab "TIMI Risk Score for UA/NSTEMI", through all 2^7 = 128 input
// combinations, and write fixtures/calculator_reference.csv with the text the app displays.
//
//   node reference/timi/query_calculator.mjs [out.csv] [settle_s]   (needs global puppeteer and /usr/bin/google-chrome)
// defaults: fixtures/calculator_reference.csv, output read once unchanged for 1 s
//
// One browser session, sequential, >= 1 s per combination. For each row we also record the 7 input values the
// Shiny server holds (Shiny.shinyapp.$inputValues) when the output is read, so a stale read can be told apart from
// an app error. The output is read once Shiny is idle and the output has not changed for 1 s.
import { createRequire } from "module";
import { execSync } from "child_process";
import { writeFileSync } from "fs";

const require = createRequire(execSync("npm root -g").toString().trim() + "/");
const puppeteer = require("puppeteer");
const URL = "https://timibiostat.shinyapps.io/calculators/";
const OUT = new globalThis.URL("fixtures/" + (process.argv[2] || "calculator_reference.csv"), import.meta.url);
const SETTLE = 2 * Number(process.argv[3] || 1);  // polls of 0.5 s
const NAMES = ["age65", "cadrisk", "knowncad", "asa", "angina", "stchange", "cardiac"];
// app inputs select_uanstemi1..7 in on-screen order: Age, >=3 risk factors, known CAD, ASA, angina, ST, markers

const b = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: true, args: ["--no-sandbox"] });
const p = await b.newPage();
await p.setUserAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36");
await p.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
// the tab link's click handler is bound only once the Shiny session is up
await p.waitForFunction(() => window.Shiny?.shinyapp?.isConnected() && document.querySelector('a.timi-link[data-tab="uanstemi"]'), { timeout: 90000 });
await new Promise(r => setTimeout(r, 2000));
await p.evaluate(() => document.querySelector('a.timi-link[data-tab="uanstemi"]').click());
await p.waitForFunction(() => document.getElementById("select_uanstemi7")?.selectize && document.getElementById("value2_uanstemi")?.innerText, { timeout: 60000 });

const rows = [["combo", ...NAMES, "server_inputs", "displayed"].join(",")];
for (let k = 0; k < 128; k++) {
  const bits = NAMES.map((_, i) => (k >> i) & 1);
  const r = await p.evaluate(async (bits, settle) => {
    const sleep = ms => new Promise(res => setTimeout(res, ms));
    bits.forEach((v, i) => document.getElementById("select_uanstemi" + (i + 1)).selectize
      .setValue(i === 0 ? (v ? ">=65" : "<65") : (v ? "Yes" : "No")));
    let prev = null, same = 0, txt = "";
    for (let t = 0; t < 60 && same < settle; t++) {
      await sleep(500);
      txt = document.getElementById("value2_uanstemi").innerText.trim();
      const busy = document.documentElement.classList.contains("shiny-busy");
      same = !busy && txt === prev ? same + 1 : 0;
      prev = txt;
    }
    const iv = Shiny.shinyapp.$inputValues;
    return { txt, server: [1, 2, 3, 4, 5, 6, 7].map(i => iv["select_uanstemi" + i]).join("|") };
  }, bits, SETTLE);
  rows.push([k, ...bits, r.server, `"${r.txt}"`].join(","));
  process.stdout.write(`${k} ${bits.join("")} ${r.txt}\n`);
}
writeFileSync(OUT, rows.join("\n") + "\n");
await b.close();
