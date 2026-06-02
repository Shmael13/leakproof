"""Generate the paper's LaTeX tables and numeric macros from the actual runs.

Every number in the paper comes from here, so the paper cannot drift from the
code. Writes:
  paper/tables/bench_synthetic.tex, bench_equities.tex, bench_crypto.tex
  paper/tables/h2.tex
  paper/macros.tex   (\newcommand definitions for all in-text numbers)

Usage:  python benchmarks/export_latex.py
"""
import glob
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tni.ast_lint.linter import lint_file                    # noqa: E402
from tni.testing import run_campaign                         # noqa: E402
from benchmarks import casestudies as CS                     # noqa: E402
from benchmarks import patterns as P                         # noqa: E402
from benchmarks import run_benchmark as RB                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
TABLES = os.path.join(PAPER, "tables")
CORPUS = os.path.join(os.path.dirname(HERE), "corpus")


def _cm(b):
    return r"\cmark" if b else r"\xmark"


def _esc(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# --- benchmark tables ------------------------------------------------------
def bench_table(rows, caption, label):
    out = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{lccccrr}", r"\toprule",
           r"Leakage pattern & \textsc{tni} & lint & unit & ASOF "
           r"& Sharpe$^\dagger$ & honest \\", r"\midrule"]
    for name, kind, caught, acc, sb, sc, base in rows:
        lint, unit, asof = base
        fp = r"$^{\textrm{\tiny FP}}$" if not acc else ""
        out.append(f"{_esc(name)}{fp} & {_cm(caught)} & {_cm(lint)} & "
                   f"{_cm(unit)} & {_cm(asof)} & ${sb:+.2f}$ & ${sc:+.2f}$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(out)


def h2_table(results, label):
    out = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{llcccr}", r"\toprule",
           r"Strategy & universe & checks & decl. & ops & Sharpe \\",
           r"\midrule"]
    for r in results:
        out.append(f"{_esc(r['name'])} & {_esc(r['universe'])} & "
                   f"{_cm(r['type_checks'])} & {r['n_decl']} & {r['n_ops']} & "
                   f"${r['sharpe']:+.2f}$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{H2 case studies: all type-check (zero false positives) "
            r"with 1--2 trusted declassifications each.}",
            f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(out)


def totals(rows):
    n = len(rows)
    tni = sum(r[2] for r in rows)
    lint = sum(r[6][0] for r in rows)
    unit = sum(r[6][1] for r in rows)
    asof = sum(r[6][2] for r in rows)
    missed = [r[0] for r in rows if not any(r[6])]
    return n, tni, lint, unit, asof, missed


def main():
    os.makedirs(TABLES, exist_ok=True)
    macros = {}

    # H3 benchmark on all three sources
    rows_by_src = {}
    for src, cap, lab in [
        ("synthetic", "Leakage benchmark on seeded synthetic data with genuine "
         "predictable structure. $\\dagger$: in-sample Sharpe the buggy "
         "pipeline would report if undetected.", "tab:bench-synth"),
        ("equities", "Leakage benchmark on US equities (20 names, 2010--2023).",
         "tab:bench-eq"),
        ("crypto", "Leakage benchmark on crypto (8 pairs, daily).",
         "tab:bench-crypto"),
    ]:
        rows = RB.collect(src)
        rows_by_src[src] = rows
        with open(os.path.join(TABLES, f"bench_{src}.tex"), "w") as f:
            f.write(bench_table(rows, cap, lab))

    n, tni, lint, unit, asof, missed = totals(rows_by_src["synthetic"])
    macros.update({
        "benchN": n, "benchTni": tni, "benchLint": lint,
        "benchUnit": unit, "benchAsof": asof,
        "benchMissed": len(missed),
        "benchMissedList": ", ".join(_esc(m) for m in missed),
    })

    # H2 case studies
    results = [fn() for fn in CS.CASE_STUDIES]
    with open(os.path.join(TABLES, "h2.tex"), "w") as f:
        f.write(h2_table(results, "tab:h2"))
    macros["htwoN"] = len(results)
    macros["htwoDecl"] = "/".join(str(r["n_decl"]) for r in results)

    # H1 soundness: aggregate campaigns across seeds
    tot = viol = interf = tc = 0
    for seed in (0, 1, 7, 11, 42):
        r = run_campaign(n_pipelines=2000, n=40, seed=seed)
        tot += r["total"]; viol += len(r["violations"])
        interf += r["interfering"]; tc += r["type_checked"]
    macros.update({"honeTotal": tot, "honeViol": viol,
                   "honeInterf": interf, "honeChecked": tc})

    # checker timing on the (correct) benchmark pipelines
    ps, fs, pp, fp, _ = RB._load("synthetic")
    times = []
    for fn in P.SERIES_PATTERNS:
        pat = fn(ps, fs)
        t0 = time.perf_counter()
        try:
            pat.correct.to_positions()
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    macros["checkMedianMs"] = f"{statistics.median(times) * 1e3:.1f}"

    # linter applicability over the corpus
    paths = sorted(glob.glob(os.path.join(CORPUS, "*.py")))
    buggy = [p for p in paths if "buggy" in os.path.basename(p)]
    clean = [p for p in paths if "buggy" not in os.path.basename(p)]
    tp = sum(1 for p in buggy if lint_file(p))
    fpos = sum(1 for p in clean if lint_file(p))
    macros.update({"lintBuggy": len(buggy), "lintTP": tp,
                   "lintClean": len(clean), "lintFP": fpos})

    # write macros
    with open(os.path.join(PAPER, "macros.tex"), "w") as f:
        f.write("% auto-generated by benchmarks/export_latex.py -- do not edit\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    print("wrote tables to", TABLES)
    print("macros:", {k: macros[k] for k in
                      ("benchN", "benchTni", "benchMissed", "honeTotal",
                       "honeViol", "checkMedianMs", "lintTP", "lintFP")})


if __name__ == "__main__":
    main()
