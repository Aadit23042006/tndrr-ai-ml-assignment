"""Build the 4-page PDF report from YOUR run results + YOUR written answers.

    python report/build_report.py --name "Your Name" --github https://github.com/you/repo \
        --hardware "Colab free T4 (16 GB), 12 GB RAM"

* Tables/numbers are read from the results/*.json files written by your own runs.
* The explanations are read from report/my_answers.md (copy my_answers.template.md and write them
  IN YOUR OWN WORDS - the company states they will ask you to explain everything live).
* The builder REFUSES to produce a final PDF if
    - any answer is missing / still says TODO / is suspiciously short,
    - any results file comes from a smoke test (--smoke),
    - the report would be longer than 4 pages even at the smallest font.
  Use --draft to build anyway (the PDF is then stamped DRAFT).
"""
import argparse
import json
import os
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ANSWER_KEYS = {
    "T1_LORA_WHY": 25, "T1_FAIL_1": 20, "T1_FAIL_2": 20, "T1_FAIL_3": 20, "T1_MEMORY_NOTES": 10,
    "T2_CHUNK_ANALYSIS": 25, "T2_FAIL_1": 20, "T2_FAIL_2": 20, "T2_BM25": 25, "T2_RERANKER": 20,
    "T3_QUALITY_OBS": 30, "T3_KV_CACHE": 25, "T3_QUANT": 25, "T3_SERVING_PLAN": 40,
}


# ------------------------------------------------------------------ loading / validation
def load_json(path, problems, required=True):
    if not os.path.exists(path):
        if required:
            problems.append(f"missing results file: {os.path.relpath(path)}")
        return None
    return json.load(open(path, encoding="utf-8"))


def parse_answers(path, problems):
    if not os.path.exists(path):
        problems.append(f"missing {os.path.relpath(path)} (copy my_answers.template.md to my_answers.md and fill it in)")
        return {}
    text = re.sub(r"<!--.*?-->", "", open(path, encoding="utf-8").read(), flags=re.S)
    parts = re.split(r"^##\s+([A-Z0-9_]+)\s*$", text, flags=re.M)
    answers = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    for key, min_words in ANSWER_KEYS.items():
        a = answers.get(key, "")
        if not a:
            problems.append(f"answer {key} is missing")
        elif "TODO" in a:
            problems.append(f"answer {key} still contains TODO")
        elif len(a.split()) < min_words:
            problems.append(f"answer {key} is too short ({len(a.split())} words, expected >= {min_words})")
    return answers


def check_not_smoke(name, obj, problems):
    if obj is not None and obj.get("smoke_test"):
        problems.append(f"{name} comes from a smoke test (--smoke), rerun it for real")


# ------------------------------------------------------------------ formatting helpers
def fmt_md(text):
    """Tiny markdown subset -> reportlab paragraph markup."""
    t = escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Mono'>\1</font>", t)
    return t.replace("\n", "<br/>")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


# ------------------------------------------------------------------ document
def build(args):
    root = args.root
    problems = []
    t1, t2, t3 = (os.path.join(root, d) for d in ("task1_lora", "task2_rag", "task3_local_inference"))
    base_s = load_json(f"{t1}/results/metrics_base_short.json", problems)
    base_i = load_json(f"{t1}/results/metrics_base_instruct.json", problems)
    ft = load_json(f"{t1}/results/metrics_finetuned.json", problems)
    abl = {r: load_json(f"{t1}/results/metrics_finetuned_r{r}.json", problems, required=False) for r in (4, 64)}
    train = load_json(f"{t1}/outputs/lora_r16/train_summary.json", problems)
    mem = load_json(f"{t1}/results/memory_calc.json", problems)
    ret = load_json(f"{t2}/results/retrieval_metrics.json", problems)
    questions = load_json(f"{t2}/questions.json", problems)
    bench = load_json(f"{t3}/results/benchmark.json", problems)
    serve = load_json(f"{t3}/results/serving_plan.json", problems)
    answers = parse_answers(os.path.join(root, "report", "my_answers.md"), problems)

    for name, o in (("metrics_base_short", base_s), ("metrics_base_instruct", base_i), ("metrics_finetuned", ft),
                    ("train_summary", train and train["summary"]), ("retrieval_metrics", ret), ("benchmark", bench)):
        check_not_smoke(name, o, problems)
    if ft and ft["overall"]["n"] < 50:
        problems.append("test set has fewer than 50 examples")
    if questions is not None and len(questions) < 15:
        problems.append(f"only {len(questions)} test questions (need 15)")
    if ret and ret.get("n_pages", 0) < 30:
        problems.append("corpus has fewer than 30 pages")
    if ret and len(ret["chunk_sizes"]) < 2:
        problems.append("need results for two chunk sizes")
    if bench and len(bench["settings"]) < 2:
        problems.append("need benchmark results for two settings")
    if not args.draft:
        for flag, label in ((args.name, "--name"), (args.github, "--github"), (args.hardware, "--hardware")):
            if not flag:
                problems.append(f"{label} is required for the final report")
    if problems:
        print("\nCannot build a FINAL report:" if not args.draft else "\nDRAFT mode - ignoring problems:")
        for p in problems:
            print("  -", p)
        if not args.draft:
            sys.exit(1)

    # fonts
    body_font, mono_font = "Helvetica", "Courier"
    dj = "/usr/share/fonts/truetype/dejavu"
    if os.path.exists(f"{dj}/DejaVuSans.ttf"):
        pdfmetrics.registerFont(TTFont("Body", f"{dj}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("Body-Bold", f"{dj}/DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("Body-Italic", f"{dj}/DejaVuSans-Oblique.ttf"))
        pdfmetrics.registerFont(TTFont("Body-BoldItalic", f"{dj}/DejaVuSans-BoldOblique.ttf"))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
        pdfmetrics.registerFont(TTFont("Mono", f"{dj}/DejaVuSansMono.ttf"))
        body_font, mono_font = "Body", "Mono"
    else:
        pdfmetrics.registerFont(pdfmetrics.Font("Mono", "Courier", "WinAnsiEncoding"))

    for fs in (9.0, 8.5, 8.0, 7.5):
        pdf_path = args.out
        n_pages = render(pdf_path, fs, body_font, mono_font, args, answers, base_s, base_i, ft, abl, train, mem,
                         ret, questions, bench, serve)
        print(f"font {fs}pt -> {n_pages} pages")
        if n_pages <= 4:
            break
    if n_pages > 4:
        msg = f"report is {n_pages} pages even at 7.5pt; shorten your answers in my_answers.md"
        if args.draft:
            print("WARNING:", msg)
        else:
            sys.exit("ERROR: " + msg)
    print(f"written {pdf_path}")


def render(pdf_path, fs, body_font, mono_font, args, answers, base_s, base_i, ft, abl, train, mem, ret,
           questions, bench, serve):
    def S(name, **kw):
        params = dict(fontName=body_font, fontSize=fs, leading=fs * 1.28, alignment=TA_LEFT)
        params.update(kw)
        return ParagraphStyle(name, **params)

    body, small = S("body", spaceAfter=2.5), S("small", fontSize=fs - 1, leading=(fs - 1) * 1.25)
    h1 = S("h1", fontName=body_font + ("-Bold" if body_font == "Body" else ""), fontSize=fs + 3.5, leading=fs + 6,
           spaceBefore=4, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
    h2 = S("h2", fontName=body_font + ("-Bold" if body_font == "Body" else ""), fontSize=fs + 1, leading=fs + 3,
           spaceBefore=4, spaceAfter=2)
    cell = S("cell", fontSize=fs - 1, leading=(fs - 1) * 1.2)

    def tbl(rows, widths=None, header=True):
        data = [[Paragraph(fmt_md(str(c)), cell) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
              ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")))
        t.setStyle(TableStyle(st))
        return t

    P = lambda text, style=body: Paragraph(fmt_md(text), style)
    A = lambda key: P(answers.get(key, "TODO"))
    W = A4[0] - 2.4 * cm
    story = []

    # ------------------------------------------------------------- header
    story.append(Paragraph(escape("tndrr AI/ML Internship - Assignment Report"), h1))
    story.append(P(f"**{args.name or '(name)'}**  |  Code: {args.github or '(github url)'}  |  Hardware: {args.hardware or '(hardware)'}", small))
    story.append(P("All numbers below come from my own runs (raw JSON files are in each task's `results/` folder).", small))

    # ------------------------------------------------------------- task 1
    story.append(Paragraph("Task 1 - LoRA fine-tuning: English calendar request to fixed JSON", h1))
    if train:
        s = train["summary"]
        story.append(P(f"**Task and data.** Model: `{s['base_model']}`. Task: turn a plain-English calendar request into JSON with keys "
                       f"title, date (YYYY-MM-DD), time (24h), duration_minutes, location, attendees. Synthetic data: "
                       f"{s['n_train_examples']} train examples; held-out test set of {ft['overall']['n'] if ft else '?'} examples "
                       f"({ft['in_dist']['n'] if ft else '?'} same-distribution + {ft['shifted']['n'] if ft else '?'} with unseen phrasings, "
                       f"date/time formats, event types and names). Metric: exact match of the whole JSON (all fields equal); "
                       f"also valid-JSON rate and per-field accuracy. Greedy decoding."))
    if base_s and base_i and ft:
        rows = [["Model / prompt", "valid JSON", "schema OK", "exact match", "exact (same-dist)", "exact (shifted)"]]
        for label, m in (("Base, short prompt", base_s), ("Base, instruction + 1-shot prompt", base_i), ("Fine-tuned (LoRA), short prompt", ft)):
            rows.append([label, pct(m["overall"]["valid_json"]), pct(m["overall"]["schema_ok"]), pct(m["overall"]["exact_match"]),
                         pct(m["in_dist"]["exact_match"]), pct(m["shifted"]["exact_match"])])
        for r in (4, 64):
            if abl.get(r):
                m = abl[r]
                rows.append([f"Fine-tuned, r={r} (ablation)", pct(m["overall"]["valid_json"]), pct(m["overall"]["schema_ok"]),
                             pct(m["overall"]["exact_match"]), pct(m["in_dist"]["exact_match"]), pct(m["shifted"]["exact_match"])])
        story += [tbl(rows, [W * 0.30, W * 0.12, W * 0.12, W * 0.13, W * 0.17, W * 0.16]), Spacer(1, 3)]
        keys = ["title", "date", "time", "duration_minutes", "location", "attendees"]
        rows = [["Field accuracy"] + keys]
        for label, m in (("Base (short)", base_s), ("Base (instruct)", base_i), ("Fine-tuned", ft)):
            rows.append([label] + [pct(m["overall"][f"field_{k}"]) for k in keys])
        story += [tbl(rows), Spacer(1, 3)]
    if train:
        l, s = train["summary"]["lora"], train["summary"]
        story.append(Paragraph("LoRA settings and why", h2))
        story.append(tbl([["rank r", "alpha", "dropout", "target modules", "lr", "epochs", "batch", "trainable params", "train time", "peak GPU mem"],
                          [l["r"], l["alpha"], l["dropout"], ", ".join(m.replace("_proj", "") for m in l["target_modules"]), l["lr"], l["epochs"],
                           l["batch_size"], f"{s['trainable_params']/1e6:.2f}M ({s['trainable_pct']:.2f}%)", f"{s['train_seconds']/60:.1f} min",
                           f"{s['peak_gpu_mem_gb']:.1f} GB" if s.get("peak_gpu_mem_gb") else "n/a"]]))
        story += [Spacer(1, 2), A("T1_LORA_WHY")]
    story.append(Paragraph("Three failures of the fine-tuned model", h2))
    for k in ("T1_FAIL_1", "T1_FAIL_2", "T1_FAIL_3"):
        story.append(A(k))
    if mem:
        b = mem["breakdown_gb"]
        story.append(Paragraph(f"GPU memory to fine-tune a 7B model (P = {mem['params']/1e9:.2f}B, LoRA r={mem['lora_rank']} = {mem['lora_params']/1e6:.0f}M trainable params)", h2))
        story.append(tbl([["Method", "Working (bytes per parameter x P)", "Total"],
                          ["(a) Full FT, mixed precision", f"bf16 weights 2P = {b['weights_bf16']:.1f} + bf16 grads 2P = {b['grads_bf16']:.1f} + Adam m,v fp32 8P = {b['adam_fp32']:.1f} + fp32 master 4P = {b['master_fp32']:.1f} + activations {mem['assumed_activation_gb']:g} GB (assumed)", f"~{mem['full_ft_gb']:.0f} GB (~{mem['full_ft_no_master_gb']:.0f} GB without fp32 master copy)"],
                          ["(b) LoRA", f"frozen bf16 weights 2P = {b['weights_bf16']:.1f} + adapter weights/grads/Adam 16 B x {mem['lora_params']/1e6:.0f}M = {b['lora_trainable_state']:.2f} + activations {mem['assumed_activation_gb']:g}", f"~{mem['lora_gb']:.0f} GB"],
                          ["(c) QLoRA", f"4-bit NF4 weights (~4.13 bit x P / 8) = {b['qlora_weights_4bit']:.1f} + adapter state {b['lora_trainable_state']:.2f} + activations {mem['assumed_activation_gb']:g}", f"~{mem['qlora_gb']:.0f} GB"]],
                         [W * 0.2, W * 0.58, W * 0.22]))
        story.append(A("T1_MEMORY_NOTES"))

    # ------------------------------------------------------------- task 2
    story.append(Paragraph("Task 2 - RAG from scratch (NumPy/FAISS, no frameworks)", h1))
    if ret:
        story.append(P(f"**Setup.** Corpus: {ret['n_docs']} public documents, {ret['n_pages']} pages. Embedding model: `{ret['embedder']}`; FAISS flat inner-product index "
                       f"over L2-normalised vectors (= cosine); BM25 implemented from scratch; hybrid = reciprocal rank fusion. Chunks are word windows with "
                       f"{int(ret['overlap_frac']*100)}% overlap. A retrieved chunk counts as a hit if it is from the right document and overlaps the gold passage by "
                       f">= 50% of the smaller of the two spans, so different chunk sizes are compared fairly."))
    if questions:
        rows = [["#", "My test question", "Source"]]
        for q in questions:
            rows.append([q.get("id", ""), q["question"][:110] + ("..." if len(q["question"]) > 110 else ""), str(q.get("doc", ""))])
        story += [Paragraph(f"{len(questions)} test questions (written by me; full gold passages in task2_rag/questions.json)", h2),
                  tbl(rows, [W * 0.05, W * 0.75, W * 0.20]), Spacer(1, 3)]
    if ret:
        ks = ret["ks"]
        rows = [["Chunk size (words)", "#chunks", "Method"] + [f"recall@{k}" for k in ks] + ["MRR"]]
        best = {}
        for size, e in ret["chunk_sizes"].items():
            for m, r in e["methods"].items():
                rows.append([size, e["n_chunks"], m] + [f"{r[f'recall@{k}']:.2f}" for k in ks] + [f"{r['mrr']:.3f}"])
                best.setdefault(m, []).append((r[f"recall@{max(ks)}"], size))
        story += [Paragraph("Retrieval accuracy", h2), tbl(rows), Spacer(1, 2)]
        story.append(P("; ".join(f"{m}: recall@{max(ks)} is higher with {max(v)[1]}-word chunks" if len({x[0] for x in v}) > 1
                                 else f"{m}: equal recall@{max(ks)} for both sizes" for m, v in best.items()) + ".", small))
        story += [A("T2_CHUNK_ANALYSIS"), Paragraph("Two retrieval failures and fixes", h2), A("T2_FAIL_1"), A("T2_FAIL_2"),
                  Paragraph("What BM25 adds, and what a re-ranker does", h2), A("T2_BM25"), A("T2_RERANKER")]

    # ------------------------------------------------------------- task 3
    story.append(Paragraph("Task 3 - Running a model locally and optimisation", h1))
    if bench:
        story.append(P(f"**Setup.** Ollama, deterministic decoding (temperature 0, seed 0), same 10 prompts, up to {bench['settings'][0]['num_predict']} new tokens, "
                       f"context {bench['settings'][0]['num_ctx']}. Tokens/s = Ollama's eval_count / eval_duration, averaged over prompts (model load time excluded). "
                       f"Hardware: {args.hardware or bench['hardware'].get('gpu') or '?'}."))
        rows = [["Setting", "mean tok/s", "min-max tok/s", "load (s)", "model memory (Ollama ps)", "GPU mem delta (nvidia-smi)", "auto-checks passed"]]
        for s in bench["settings"]:
            m = s["memory"]
            rows.append([f"`{s['model']}`", num(s["mean_tokens_per_s"]), f"{num(s['min_tokens_per_s'])}-{num(s['max_tokens_per_s'])}", num(s["load_seconds"]),
                         f"{num(m.get('ollama_ps_total_gb'), 2)} GB", f"{num(m['nvidia_smi_delta_mib'], 0)} MiB" if m.get("nvidia_smi_delta_mib") is not None else "n/a",
                         f"{s['auto_checks_passed']}/{s['auto_checks_total']}"])
        story += [tbl(rows), Spacer(1, 2), Paragraph("Output quality on the same 10 prompts", h2), A("T3_QUALITY_OBS"),
                  P("Full side-by-side outputs: task3_local_inference/results/quality_comparison.md.", small)]
    story += [Paragraph("KV cache and quantisation (in my own words)", h2), A("T3_KV_CACHE"), A("T3_QUANT")]
    if serve:
        a = serve["assumptions"]
        story.append(Paragraph(f"Serving {a['users']} concurrent users on one {a['gpu_gb']:g} GB GPU: back-of-envelope", h2))
        story.append(P(f"**Assumptions.** ~{a['params_b']}B-parameter model ({a['layers']} layers, {a['kv_heads']} KV heads, head_dim {a['head_dim']}), "
                       f"{a['ctx']} tokens per user, {int(a['util']*100)}% of GPU memory usable, {a['overhead_gb']:g} GB overhead, memory bandwidth {a['bw_gbs']:g} GB/s at "
                       f"{int(a['eff']*100)}% efficiency, {a['tflops']:g} TFLOPs at {int(a['mfu']*100)}% utilisation for prefill (prompt {a['prompt_tokens']} tokens)."))
        rows = [["Scenario", "weights", "KV (all users)", "total / usable", "fits?", "max users", "decode tok/s per user", "prefill / prompt"]]
        for s in serve["scenarios"]:
            rows.append([s["scenario"], f"{s['weights_gb']:.1f} GB", f"{s['kv_all_users_gb']:.1f} GB", f"{s['total_gb']:.1f} / {s['budget_gb']:.1f} GB",
                         "yes" if s["fits"] else "no", s["max_users_by_memory"], f"{s['per_user_tokens_per_s']:.0f}", f"{s['prefill_s_per_prompt']:.2f} s"])
        story += [tbl(rows, [W * 0.30, W * 0.09, W * 0.11, W * 0.14, W * 0.06, W * 0.08, W * 0.12, W * 0.10]), Spacer(1, 2), A("T3_SERVING_PLAN")]

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(body_font, 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.2 * cm, 0.7 * cm, f"tndrr AI/ML assignment - {args.name or ''}")
        canvas.drawRightString(A4[0] - 1.2 * cm, 0.7 * cm, f"page {doc.page}")
        if args.draft:
            canvas.setFillColor(colors.red)
            canvas.setFont(body_font, 9)
            canvas.drawCentredString(A4[0] / 2, A4[1] - 0.7 * cm, "DRAFT - NOT FOR SUBMISSION")
        canvas.restoreState()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.1 * cm,
                            bottomMargin=1.2 * cm, title="tndrr AI/ML Assignment Report", author=args.name or "")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    from pypdf import PdfReader
    return len(PdfReader(pdf_path).pages)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(here), help="repository root")
    ap.add_argument("--out", default=os.path.join(here, "report.pdf"))
    ap.add_argument("--name", default="")
    ap.add_argument("--github", default="")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--draft", action="store_true")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
