"""Run the (unsound, best-effort) AST linter over the corpus and report how
widespread look-ahead bias is (DESIGN.md B.8).

Usage:  python benchmarks/run_linter.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tni.ast_lint.linter import lint_file       # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus")


def main():
    paths = sorted(glob.glob(os.path.join(CORPUS, "*.py")))
    lines = ["# AST linter results (best-effort, UNSOUND)\n",
             "This linter pattern-matches syntactic red flags in ordinary "
             "pandas/sklearn code. It has false positives and false negatives "
             "by design; it does not affect the soundness theorem. Ground truth "
             "is taken from the `_buggy`/`_clean` filename convention.\n"]
    tp = fp = buggy = clean = 0
    for p in paths:
        findings = lint_file(p)
        name = os.path.basename(p)
        truth_buggy = "buggy" in name
        buggy += truth_buggy
        clean += (not truth_buggy)
        if findings:
            tp += truth_buggy
            fp += (not truth_buggy)
            lines.append(f"\n**{name}** ({'buggy' if truth_buggy else 'clean'})"
                         f" — {len(findings)} finding(s):")
            for f in findings:
                lines.append(f"  - L{f.line} [{f.code}] {f.message}")
        else:
            lines.append(f"\n**{name}** ({'buggy' if truth_buggy else 'clean'})"
                         f" — no findings")
    lines += [
        f"\nRecall on buggy scripts: {tp}/{buggy} flagged.",
        f"False positives among clean scripts: {fp}/{clean} flagged.",
        "\nThe false positive is instructive: the linter flags any negative "
        "`.shift()`, but a forward-shifted *target* (y = next return) is "
        "legitimate -- a syntactic tool cannot tell a leaked feature from a "
        "label. The sound tni core has no such false positive (cf. H2: 3/3 "
        "correct strategies type-check), which is exactly why the core, not the "
        "linter, carries the guarantee.",
    ]
    text = "\n".join(lines)
    print(text)
    out = os.path.join(os.path.dirname(__file__), "LINT_RESULTS.md")
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
