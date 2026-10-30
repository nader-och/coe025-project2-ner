// Build presentation/project2.pptx (title + 5 slides, 2-minute talk).
// Every number on the slides is read from files in results/ - nothing is typed by hand.
// Not in slides: needs Node.js and `npm install pptxgenjs@3.12.0`.
// Run from the project root:  node presentation/make_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const read = (...p) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", ...p), "utf8"));
const test = read("test_results.json").results;
const timing = read("timing", "timing_colab.json").methods;
const data = read("data_inspection.json");
const err = read("error_analysis_test.json");

const f3 = (x) => x.toFixed(3);
const ms = (x) => (x < 0.1 ? x.toFixed(3) : x < 10 ? x.toFixed(2) : x.toFixed(0));
const pct = (x) => (100 * x).toFixed(1) + "%";
const n = (x) => x.toLocaleString("en-US");

const THEME = {
  name: "BC5CDR NER",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A2F", lt1: "FFFFFF", dk2: "0B4F5C", lt2: "E8F3F1",
    accent1: "0B7A75",   // Chemical (teal)
    accent2: "D9572B",   // Disease (coral)
    accent3: "6B7C80",   // O / muted
    accent4: "1E8C4E",   // correct
    accent5: "C0392B",   // wrong
    accent6: "F2B134",
    hlink: "0B7A75", folHlink: "0B4F5C",
  },
};
const T = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Finding Chemicals and Diseases in Medical Text";
pres.author = "Çınar Sabancı, Bashir Alzoubi, Nader Ochen";
pres.company = "Çınar Sabancı, Bashir Alzoubi, Nader Ochen";
pres.subject = "COE025 NLP Project 2: Named Entity Recognition on BC5CDR";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: T.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.5,
        fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" },
      text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.6,
        fontSize: 20, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: T.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
        fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left" },
      text: "" } },
    { text: { text: "COE025 NLP · Project 2 · NER on BC5CDR",
        options: { x: 0.6, y: 6.95, w: 6, h: 0.35, fontSize: 10, color: C.accent3 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.35, fontSize: 10, color: C.accent3 },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: T.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
        fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left" },
      text: "" } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.35, fontSize: 10, color: C.background2 },
});

const LABEL_COLOR = (tag) => (tag.endsWith("Chemical") ? C.accent1 : tag.endsWith("Disease") ? C.accent2 : C.accent3);

// A row of word "chips" with their BIO label underneath (Week 6, Slide 23).
function tokenRow(slide, tokens, tags, x0, y, name) {
  let x = x0;
  tokens.forEach((w, i) => {
    const width = 0.4 + 0.135 * Math.max(w.length, tags[i].length * 0.75);
    const isEnt = tags[i] !== "O";
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: width, h: 0.75, rectRadius: 0.08,
      fill: { color: isEnt ? C.background2 : C.background1 },
      line: { color: isEnt ? LABEL_COLOR(tags[i]) : C.accent3, width: isEnt ? 2 : 0.75 },
      objectName: `${name}-chip-${i}` });
    slide.addText(w, { x, y, w: width, h: 0.75, align: "center", valign: "middle", fontSize: 18,
      bold: isEnt, color: C.text1, margin: 0, isTextBox: true, objectName: `${name}-word-${i}` });
    slide.addText(tags[i], { x: x - 0.1, y: y + 0.82, w: width + 0.2, h: 0.4, align: "center",
      fontSize: 14, bold: isEnt, color: LABEL_COLOR(tags[i]), margin: 0, isTextBox: true,
      objectName: `${name}-tag-${i}` });
    x += width + 0.15;
  });
}

function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: fill, width: 0 }, objectName: name });
}

// ---------- Title ----------
pres.addSection({ title: "Title" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
s.addText("Finding Chemicals and Diseases in Medical Text", { placeholder: "title" });
s.addText([
  { text: "Named Entity Recognition on BC5CDR: five methods compared", options: { breakLine: true } },
  { text: "COE025 Natural Language Processing · Project 2", options: { fontSize: 16, breakLine: true } },
  { text: "Çınar Sabancı (2409115637) · Bashir Alzoubi (2409115364) · Nader Ochen (2309116429)",
    options: { fontSize: 16 } },
], { placeholder: "body" });
s.addNotes("[0:00-0:05] Hello. Our project finds chemical and disease names in medical text.");

// ---------- 1. Problem ----------
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Problem: find chemical and disease names", { placeholder: "title" });
s.addText("Named Entity Recognition (NER): give every word a BIO label", { x: 0.6, y: 1.35, w: 12.1, h: 0.5,
  fontSize: 20, color: C.text1, isTextBox: true, objectName: "task-line" });
// A real TRAINING sentence (line 210 of data/raw/dataset/train.json) with B and I labels of both types.
const id2label = Object.fromEntries(Object.entries(JSON.parse(fs.readFileSync(
  path.join(ROOT, "data", "raw", "dataset", "label.json"), "utf8"))).map(([l, i]) => [i, l]));
const rawEx = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "raw", "dataset", "train.json"), "utf8")
  .split(/\r?\n/).filter((l) => l.trim())[209]);
const ex = { tokens: rawEx.tokens, tags: rawEx.tags.map((t) => id2label[t]) };
tokenRow(s, ex.tokens, ex.tags, 0.6, 2.15, "example");
s.addText("Real training sentence from BC5CDR", { x: 0.6, y: 3.45, w: 6, h: 0.35, fontSize: 12,
  italic: true, color: C.accent3, isTextBox: true, objectName: "example-caption" });

const sp = data.splits;
const cards = [
  ["B / I / O", "B = first word of a name · I = next word · O = outside"],
  ["2 types → 5 labels", "B- and I- for Chemical and for Disease, plus O"],
  [`${n(sp.train.sentences + sp.validation.sentences + sp.test.sentences)} sentences`,
    `BC5CDR: ${n(sp.train.sentences)} train · ${n(sp.validation.sentences)} validation · ${n(sp.test.sentences)} test`],
];
cards.forEach(([head, body], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 4.1, 3.9, 1.75, `info-card-${i}`);
  s.addText(head, { x: x + 0.25, y: 4.25, w: 3.4, h: 0.6, fontSize: 22, bold: true, color: C.text2,
    margin: 0, isTextBox: true, objectName: `info-head-${i}` });
  s.addText(body, { x: x + 0.25, y: 4.85, w: 3.4, h: 0.85, fontSize: 15, color: C.text1, margin: 0,
    valign: "top", isTextBox: true, objectName: `info-body-${i}` });
});
s.addText("Real-world use: medical records (Week 1, Slide 31)", { x: 0.6, y: 6.1, w: 12.1, h: 0.45,
  fontSize: 16, color: C.text1, isTextBox: true, objectName: "use-line" });
s.addNotes("[0:05-0:30] This task is called Named Entity Recognition. We give every word a label. " +
  "B means the first word of a name, I means the next words, and O means outside. " +
  "We use the BC5CDR dataset from Hugging Face. It has about sixteen thousand sentences from medical articles.");

// ---------- 2. Methods ----------
pres.addSection({ title: "Methods" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methods" });
s.addText("Five methods, from simple to strong", { placeholder: "title" });
const methods = [
  ["Frequent-tag baseline", "Most frequent training label for each word (no context)"],
  ["HMM + Viterbi", "Adds label transitions; Viterbi finds the best whole label path"],
  ["Feature CRF", "Adds rich features: word, suffix, capital letters, neighbor words"],
  ["BERT + softmax", "Adds context from the whole sentence (pre-trained BERT)"],
  ["BERT + CRF", "BERT scores + CRF: the output always follows the BIO rules"],
];
methods.forEach(([name, adds], i) => {
  const y = 1.45 + i * 0.95;
  s.addShape(pres.shapes.OVAL, { x: 0.7, y: y + 0.08, w: 0.62, h: 0.62,
    fill: { color: i === 4 ? C.accent1 : C.text2 }, line: { color: C.background1, width: 0 },
    objectName: `method-dot-${i}` });
  s.addText(String(i + 1), { x: 0.7, y: y + 0.08, w: 0.62, h: 0.62, align: "center", valign: "middle",
    fontSize: 20, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: `method-num-${i}` });
  s.addText(name, { x: 1.6, y, w: 3.6, h: 0.78, valign: "middle", fontSize: 20, bold: true,
    color: C.text1, margin: 0, isTextBox: true, objectName: `method-name-${i}` });
  s.addText(adds, { x: 5.3, y, w: 7.4, h: 0.78, valign: "middle", fontSize: 17, color: C.text1,
    margin: 0, isTextBox: true, objectName: `method-adds-${i}` });
});
card(s, 0.6, 6.25, 12.1, 0.55, "protocol-card");
s.addText("Same split for all methods · settings chosen on validation · test set used once", {
  x: 0.85, y: 6.25, w: 11.6, h: 0.55, valign: "middle", fontSize: 15, color: C.text2, margin: 0,
  isTextBox: true, objectName: "protocol-line" });
s.addNotes("[0:30-0:55] We compared five methods, from simple to strong. The baseline gives each word its " +
  "most frequent label. The HMM adds label transitions and Viterbi. The CRF adds features like suffixes. " +
  "BERT adds context from the whole sentence. BERT plus CRF also keeps the BIO rules. " +
  "All methods use the same split, and we used the test set only once.");

// ---------- 3. Results ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results on the test set", { placeholder: "title" });
const keys = ["baseline", "hmm", "crf", "bert_softmax", "bert_crf"];
const names = { baseline: "1. Frequent-tag baseline", hmm: "2. HMM + Viterbi", crf: "3. Feature CRF",
  bert_softmax: "4. BERT + softmax", bert_crf: "5. BERT + CRF" };
const best = keys.reduce((a, b) => (test[b].micro.f1 > test[a].micro.f1 ? b : a));
const head = ["Method", "Entity F1", "Chemical F1", "Disease F1", "ms / sentence\nCPU", "ms / sentence\nGPU"]
  .map((t) => ({ text: t, options: { bold: true, color: T.lt1, fill: { color: T.dk2 }, align: "center", valign: "middle" } }));
const rows = [head];
keys.forEach((k) => {
  const isBest = k === best;
  const o = { fill: { color: isBest ? "D5ECE8" : T.lt1 }, bold: isBest, color: T.dk1, align: "center", valign: "middle" };
  const gpu = timing[`${k}_gpu`];
  rows.push([
    { text: names[k], options: { ...o, align: "left" } },
    { text: f3(test[k].micro.f1), options: { ...o, fontSize: 18 } },
    { text: f3(test[k].per_type.Chemical.f1), options: o },
    { text: f3(test[k].per_type.Disease.f1), options: o },
    { text: ms(timing[`${k}_cpu`].ms_per_sentence), options: o },
    { text: gpu ? ms(gpu.ms_per_sentence) : "–", options: o },
  ]);
});
s.addTable(rows, { x: 0.6, y: 1.45, w: 12.1, colW: [3.6, 1.7, 1.7, 1.7, 1.7, 1.7], rowH: 0.62,
  fontSize: 16, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.75, color: "C9D6D4" },
  objectName: "results-table" });
s.addText([
  { text: "Entity F1: exact match (start, end and type must all be correct). ", options: { breakLine: true } },
  { text: "Speed: same Google Colab machine; BERT is fast only on a GPU." },
], { x: 0.6, y: 5.55, w: 12.1, h: 0.8, fontSize: 14, color: C.accent3, isTextBox: true, objectName: "results-note" });
s.addNotes("[0:55-1:20] Here are the test results. The score is entity F1 with exact match. " +
  `BERT plus CRF is the best, with ${f3(test.bert_crf.micro.f1)}. The baseline gets ${f3(test.baseline.micro.f1)}. ` +
  "Disease is harder than chemical for every method. " +
  `But BERT is slow on a CPU: about ${ms(timing.bert_crf_cpu.ms_per_sentence)} milliseconds per sentence.`);

// ---------- 4. Key insight ----------
pres.addSection({ title: "Insight" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Insight" });
s.addText("Key insights", { placeholder: "title" });
// Left: accuracy misleads (Week 6, Slide 36)
card(s, 0.6, 1.45, 4.6, 4.9, "accuracy-card");
s.addText("Accuracy misleads", { x: 0.9, y: 1.65, w: 4.0, h: 0.5, fontSize: 22, bold: true, color: C.text2,
  margin: 0, isTextBox: true, objectName: "acc-head" });
s.addText("Predict O for every word:", { x: 0.9, y: 2.3, w: 4.0, h: 0.45, fontSize: 16, color: C.text1,
  margin: 0, isTextBox: true, objectName: "acc-sub" });
s.addText(pct(test.all_O.token_accuracy), { x: 0.9, y: 2.8, w: 4.0, h: 1.0, fontSize: 60, bold: true,
  color: C.text2, margin: 0, isTextBox: true, objectName: "acc-big" });
s.addText("token accuracy", { x: 0.9, y: 3.8, w: 4.0, h: 0.4, fontSize: 15, color: C.accent3, margin: 0,
  isTextBox: true, objectName: "acc-label" });
s.addText(f3(test.all_O.micro.f1), { x: 0.9, y: 4.4, w: 4.0, h: 1.0, fontSize: 60, bold: true,
  color: C.accent5, margin: 0, isTextBox: true, objectName: "f1-big" });
s.addText("entity F1: no names found", { x: 0.9, y: 5.4, w: 4.0, h: 0.4, fontSize: 15, color: C.accent3,
  margin: 0, isTextBox: true, objectName: "f1-label" });

// Right: structured decoding fixes invalid BIO (Week 6, Slides 23, 80, 87)
card(s, 5.5, 1.45, 7.2, 4.9, "bio-card");
s.addText("Softmax breaks the BIO rules", { x: 5.8, y: 1.65, w: 6.6, h: 0.5, fontSize: 22, bold: true,
  color: C.text2, margin: 0, isTextBox: true, objectName: "bio-head" });
const exm = err.examples.find((e) => e.rule.startsWith("BERT + softmax invalid"));
const shown = exm.tokens.slice(0, 4);
const tagRow = (label, tags, mark, color) => [
  { text: label, options: { bold: true, align: "left", color: T.dk1 } },
  ...tags.slice(0, 4).map((t) => ({ text: t, options: { color: t.endsWith("Chemical") ? T.accent1 : T.accent3,
    bold: t !== "O" } })),
  { text: mark, options: { bold: true, color } },
];
const bioRows = [
  [{ text: "", options: {} }, ...shown.map((w) => ({ text: w, options: { bold: true, color: T.dk1 } })),
    { text: "", options: {} }],
  tagRow("Gold", exm.gold, "", T.dk1),
  tagRow("BERT + softmax", exm.predictions.bert_softmax, "✗", T.accent5),
  tagRow("BERT + CRF", exm.predictions.bert_crf, "✓", T.accent4),
];
s.addTable(bioRows, { x: 5.65, y: 2.35, w: 6.95, colW: [1.8, 0.7, 1.35, 1.3, 1.4, 0.4], rowH: 0.5,
  fontSize: 16, fontFace: THEME.bodyFontFace, align: "center", valign: "middle",
  border: { type: "none" }, fill: { color: T.lt2 }, objectName: "bio-table" });
s.addText("O → I-Chemical is not allowed: an I must follow a B", { x: 5.8, y: 4.5, w: 6.6, h: 0.45,
  fontSize: 15, color: C.text1, margin: 0, isTextBox: true, objectName: "bio-rule" });
s.addText(`Invalid label pairs on the test set: softmax ${test.bert_softmax.invalid_transitions} · BERT + CRF ${test.bert_crf.invalid_transitions}`,
  { x: 5.8, y: 5.15, w: 6.6, h: 0.9, fontSize: 18, bold: true, color: C.text2, margin: 0, valign: "top",
    isTextBox: true, objectName: "bio-count" });
s.addNotes("[1:20-1:45] Two lessons. First, accuracy misleads: if we predict O for every word, accuracy is " +
  `${(100 * test.all_O.token_accuracy).toFixed(0)} percent, but we find zero entities. Second, BERT with softmax ` +
  "breaks the BIO rules: here it starts with I-Chemical. The CRF never does this.");

// ---------- 5. Conclusion ----------
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
s.addText("Conclusion: the right tool for the right job", { placeholder: "title" });
const concl = [
  ["With a GPU", "BERT + CRF", [`Best entity F1: ${f3(test.bert_crf.micro.f1)}`, "Always valid BIO labels",
    `${ms(timing.bert_crf_gpu.ms_per_sentence)} ms per sentence on a GPU`]],
  ["CPU only", "Feature CRF", [`Entity F1: ${f3(test.crf.micro.f1)}`, "Always valid BIO labels",
    `${ms(timing.crf_cpu.ms_per_sentence)} ms per sentence (BERT on CPU: ${ms(timing.bert_crf_cpu.ms_per_sentence)} ms)`]],
];
concl.forEach(([when, pick, lines], i) => {
  const x = 0.6 + i * 6.2;
  card(s, x, 1.5, 5.9, 3.2, `concl-card-${i}`, C.background1);
  s.addText(when, { x: x + 0.35, y: 1.75, w: 5.2, h: 0.45, fontSize: 18, color: C.accent3, margin: 0,
    isTextBox: true, objectName: `concl-when-${i}` });
  s.addText(pick, { x: x + 0.35, y: 2.25, w: 5.2, h: 0.75, fontSize: 32, bold: true, color: C.text2,
    margin: 0, isTextBox: true, objectName: `concl-pick-${i}` });
  s.addText(lines.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < lines.length - 1 } })),
    { x: x + 0.35, y: 3.15, w: 5.3, h: 2.0, fontSize: 17, color: C.text1, paraSpaceAfter: 6, valign: "top",
      isTextBox: true, objectName: `concl-lines-${i}` });
});
s.addText([{ text: "Code: " }, { text: "https://github.com/nader-och/coe025-project2-ner",
  options: { hyperlink: { url: "https://github.com/nader-och/coe025-project2-ner" }, color: C.background1 } }], { x: 0.6, y: 5.1, w: 12.1, h: 0.5,
  fontSize: 18, color: C.background1, isTextBox: true, objectName: "github" });
s.addNotes("[1:45-2:00] So, with a GPU we choose BERT plus CRF. On a CPU only, we choose the feature CRF: " +
  "it is fast and good. The code is on GitHub. Thank you.");

(async () => {
  const out = path.join(__dirname, "project2.pptx");
  await pres.writeFile({ fileName: out });
  // Writes the theme colors into the file (helper from the pptx tool used to build this deck).
  if (process.env.APPLY_THEME) await require(process.env.APPLY_THEME).applyTheme(out, THEME);
  else console.log("APPLY_THEME not set: theme colors not written (slides still use the hex colors)");
  console.log("wrote", out);
})();
