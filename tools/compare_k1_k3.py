"""
compare_k1_k3.py
================
Runs brute_force_k3.py (3-word shingles) and brute_force_k1.py (single words)
on the same folder of documents and compares their results.

Usage (from the project folder):
    python compare_k1_k3.py Document
    python compare_k1_k3.py Document --threshold 0.2 --out comparison_output

Needs in the same folder: brute_force_k3.py, brute_force_k1.py,
document_loader.py, extract_books.py, text_quality.py

Shows:
  1. size of each document's shingle set under k=3 and k=1
  2. every pair side by side: shared / union / Jaccard for both, and the rank of the pair
  3. summary statistics and how many pairs pass each threshold
  4. how well the two methods agree (rank correlation, top pairs)
  5. which pairs are called "similar" by one method but not the other
  6. each document as a query: top matches under both
  7. timing

Files written to --out (default comparison_output):
  comparison_report.txt, comparison_k1_vs_k3.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import brute_force_k1 as b1
import brute_force_k3 as b3

LINES: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    LINES.append(line)


def average_ranks(values):
    """Rank 1 = largest value; tied values share the average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float("nan")


def biggest_gap(values):
    """Largest drop between neighbouring scores when sorted high to low."""
    v = sorted(values, reverse=True)
    best = (0.0, 0)
    for i in range(len(v) - 1):
        if v[i] - v[i + 1] > best[0]:
            best = (v[i] - v[i + 1], i)
    return best[0], v[best[1]], v[best[1] + 1], best[1] + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="Document")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="score above which a pair is called 'similar' in section 5 (default 0.2)")
    ap.add_argument("--out", default="comparison_output")
    args = ap.parse_args()
    if not Path(args.folder).is_dir():
        raise SystemExit(f"Folder not found: {Path(args.folder).resolve()}")

    r3 = b3.analyse(args.folder)
    r1 = b1.analyse(args.folder)
    names = r3["names"]
    n = len(names)

    p3 = {(p["a"], p["b"]): p for p in r3["pairs"]}
    p1 = {(p["a"], p["b"]): p for p in r1["pairs"]}
    keys = list(p3)
    j3 = [p3[k]["jaccard"] for k in keys]
    j1 = [p1[k]["jaccard"] for k in keys]
    rank3 = average_ranks(j3)
    rank1 = average_ranks(j1)

    emit("COMPARISON: brute force with k = 3 (3-word phrases) vs k = 1 (single words)")
    emit(f"Folder: {Path(args.folder).resolve()}   documents: {n}   pairs: {len(keys)}\n")

    # ---- 1. set sizes -------------------------------------------------
    emit("1. Size of each document's shingle set")
    emit(f"{'#':>3} {'document':<14}{'words':>8}{'k=3 shingles':>14}{'k=1 shingles':>14}")
    for i, name in enumerate(names, 1):
        emit(f"{i:>3} {name[:14]:<14}{r3['words'][name]:>8,}"
             f"{r3['shingle_counts'][name]:>14,}{r1['shingle_counts'][name]:>14,}")
    emit("k=3 counts distinct 3-word phrases; k=1 counts distinct words (the vocabulary).\n")

    # ---- 2. pair table ------------------------------------------------
    order = sorted(range(len(keys)), key=lambda i: (-j3[i], -j1[i]))
    emit("2. Every pair side by side (sorted by the k=3 score)")
    emit(f"{'pair':<20}| {'shared':>6}{'union':>7}{'J(k=3)':>8} | "
         f"{'shared':>6}{'union':>7}{'J(k=1)':>8} | {'diff':>7} | {'rank3':>5}{'rank1':>6}")
    for i in order:
        k = keys[i]
        a, b = p3[k], p1[k]
        label = f"{k[0][:9]} / {k[1][:8]}"
        emit(f"{label:<20}| {a['shared']:>6}{a['union']:>7}{a['jaccard']:>8.3f} | "
             f"{b['shared']:>6}{b['union']:>7}{b['jaccard']:>8.3f} | "
             f"{b['jaccard'] - a['jaccard']:>+7.3f} | {rank3[i]:>5.1f}{rank1[i]:>6.1f}")
    emit("diff = J(k=1) - J(k=3). rank 1 = most similar pair; tied pairs share the average rank.\n")

    # ---- 3. summary ---------------------------------------------------
    emit("3. Summary over all pairs")
    emit(f"{'':<28}{'k=3':>10}{'k=1':>10}")
    for label, fn in (("mean", statistics.mean), ("median", statistics.median),
                      ("maximum", max), ("minimum", min)):
        emit(f"{label:<28}{fn(j3):>10.4f}{fn(j1):>10.4f}")
    emit(f"{'pairs scoring exactly 0':<28}{sum(v == 0 for v in j3):>10}{sum(v == 0 for v in j1):>10}")
    for t in (0.8, 0.5, 0.2, 0.05):
        emit(f"{'pairs with Jaccard >= ' + str(t):<28}{sum(v >= t for v in j3):>10}"
             f"{sum(v >= t for v in j1):>10}")
    g3, g1 = biggest_gap(j3), biggest_gap(j1)
    emit(f"{'largest gap between ranked scores':<28}"
         f"{g3[0]:>10.3f}{g1[0]:>10.3f}")
    emit(f"   k=3: the scores drop from {g3[1]:.3f} to {g3[2]:.3f} after the top {g3[3]} pairs")
    emit(f"   k=1: the scores drop from {g1[1]:.3f} to {g1[2]:.3f} after the top {g1[3]} pairs\n")

    # ---- 4. agreement -------------------------------------------------
    emit("4. Do the two methods agree?")
    emit(f"  Spearman rank correlation of the {len(keys)}-pair rankings: {pearson(rank3, rank1):.3f}"
         if len(keys) > 2 else "  (too few pairs)")
    top_n = min(6, len(keys))
    top3 = {keys[i] for i in sorted(range(len(keys)), key=lambda i: (-j3[i], keys[i]))[:top_n]}
    top1 = {keys[i] for i in sorted(range(len(keys)), key=lambda i: (-j1[i], keys[i]))[:top_n]}
    emit(f"  Pairs shared by the two top-{top_n} lists: {len(top3 & top1)} of {top_n}")
    emit("  (1.000 = identical ordering; values near 0 = unrelated ordering)\n")

    # ---- 5. decisions at a threshold ---------------------------------
    t = args.threshold
    both = [k for k, a, b in zip(keys, j3, j1) if a >= t and b >= t]
    only3 = [k for k, a, b in zip(keys, j3, j1) if a >= t > b]
    only1 = [k for k, a, b in zip(keys, j3, j1) if b >= t > a]
    emit(f"5. Pairs called 'similar' at threshold {t}")
    emit(f"  called similar by both methods : {len(both)}")
    emit(f"  called similar by k=3 only     : {len(only3)}")
    emit(f"  called similar by k=1 only     : {len(only1)}")
    for k in only1:
        emit(f"      k=1 only: {k[0]} / {k[1]}   J(k=3)={p3[k]['jaccard']:.3f}   J(k=1)={p1[k]['jaccard']:.3f}")
    for k in only3:
        emit(f"      k=3 only: {k[0]} / {k[1]}   J(k=3)={p3[k]['jaccard']:.3f}   J(k=1)={p1[k]['jaccard']:.3f}")
    emit()

    # ---- 6. queries ---------------------------------------------------
    emit("6. Each document as a query: top 3 matches")
    same_top1 = 0
    for name in names:
        t3, t1 = r3["queries"][name], r1["queries"][name]
        same_top1 += bool(t3 and t1 and t3[0][0] == t1[0][0])
        emit(f"  {name}")
        emit("      k=3: " + ", ".join(f"{d} ({s:.3f})" for d, s in t3))
        emit("      k=1: " + ", ".join(f"{d} ({s:.3f})" for d, s in t1))
    emit(f"  The best match is the same document under both methods for {same_top1} of {n} queries.\n")

    # ---- 7. timing ----------------------------------------------------
    a, b = r3["timing"], r1["timing"]
    emit("7. Timing (this machine, this run)")
    emit(f"{'':<34}{'k=3':>12}{'k=1':>12}")
    emit(f"{'build shingle sets (ms)':<34}{a['build_seconds'] * 1000:>12.2f}{b['build_seconds'] * 1000:>12.2f}")
    emit(f"{'one comparison, median (us)':<34}{a['median_comparison_us']:>12.1f}{b['median_comparison_us']:>12.1f}")
    emit(f"{'all ' + str(a['comparisons']) + ' comparisons (ms)':<34}"
         f"{a['all_pairs_seconds'] * 1000:>12.3f}{b['all_pairs_seconds'] * 1000:>12.3f}")
    avg3 = statistics.mean(r3["shingle_counts"].values())
    avg1 = statistics.mean(r1["shingle_counts"].values())
    emit(f"{'average set size':<34}{avg3:>12.0f}{avg1:>12.0f}")
    emit("A comparison costs about as much as the size of the smaller set, so the "
         "smaller k=1 sets compare faster.")

    # ---- save ---------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison_k1_vs_k3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_a", "doc_b", "shared_k3", "union_k3", "jaccard_k3",
                    "shared_k1", "union_k1", "jaccard_k1", "diff_k1_minus_k3",
                    "rank_k3", "rank_k1"])
        for i in order:
            k = keys[i]
            w.writerow([k[0], k[1], p3[k]["shared"], p3[k]["union"], f"{j3[i]:.6f}",
                        p1[k]["shared"], p1[k]["union"], f"{j1[i]:.6f}",
                        f"{j1[i] - j3[i]:.6f}", rank3[i], rank1[i]])
    (out / "comparison_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nFiles written to: {out.resolve()}")


if __name__ == "__main__":
    main()
