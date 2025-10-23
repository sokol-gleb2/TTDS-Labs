#!/usr/bin/env python3
"""
TTDS Lab 1 helper script

Covers:
- Preprocessing (tokenisation, case-folding, stopword removal, stemming)
- Printing preprocessed collections
- Zipf's law (rank vs frequency, log–log)
- Benford's law on term frequencies (with/without single digits)
- Heap's law (vocabulary growth) + fit V = k * N^b

Usage examples:

python ttds_lab1.py \
  --collections bible.txt quran.txt abstracts.wiki.txt \
  --outdir outputs

# Optional custom stopword file (one word per line):
python ttds_lab1.py --stopwords path/to/stopwords.txt \
  --collections bible.txt quran.txt abstracts.wiki.txt

Notes:
- Requires Python 3, numpy, matplotlib, nltk (no corpora downloads needed).
- Gzip files are supported (.gz).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import gzip
import io
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

# ----------- Defaults -----------
# Minimal English stopword list (extend as needed or pass a file)
DEFAULT_STOPWORDS = {
    'a','an','and','are','as','at','be','but','by','for','if','in','into','is','it','no','not','of',
    'on','or','such','that','the','their','then','there','these','they','this','to','was','will','with',
    'i','me','my','we','our','you','your','he','him','his','she','her','hers','them','their','theirs',
    'who','whom','which','what','when','where','why','how','from','than','too','very','can','could',
}

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

@dataclass
class CorpusStats:
    term_freq: Counter
    heaps_points: List[Tuple[int, int]]  # (N tokens seen, V vocab size)

# ----------- IO helpers -----------

def smart_open(path: str) -> io.TextIOBase:
    return open(path, 'r', encoding='utf-8', errors='ignore')

# ----------- Preprocessing -----------

def iter_tokens(stream: Iterable[str]) -> Iterator[str]:
    for line in stream:
        for m in TOKEN_RE.finditer(line.lower()):
            yield m.group(0)


def preprocess_tokens(tokens: Iterable[str], stopwords: set[str], stemmer: PorterStemmer) -> Iterator[str]:
    for tok in tokens:
        if tok in stopwords:
            continue
        yield stemmer.stem(tok)


def write_preprocessed(input_path: str, out_path: str, stopwords: set[str]) -> Counter:
    stemmer = PorterStemmer()
    tf = Counter()
    with smart_open(input_path) as f, open(out_path, 'w', encoding='utf-8') as out:
        for tok in preprocess_tokens(iter_tokens(f), stopwords, stemmer):
            tf[tok] += 1
            out.write(tok + ' ')
    return tf

# ----------- Zipf -----------

def plot_zipf(term_freq: Counter, title: str, out_path: str):
    freqs = np.array(sorted(term_freq.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    plt.figure()
    plt.loglog(ranks, freqs, marker='.', linestyle='none')
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title(f"Zipf's law – {title}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ----------- Benford -----------

def first_digit(n: int) -> int:
    while n >= 10:
        n //= 10
    return n


def benford_distribution(counts: Iterable[int], drop_lt_10: bool) -> Dict[int, float]:
    c = Counter()
    total = 0
    for v in counts:
        if v <= 0:
            continue
        if drop_lt_10 and v < 10:
            continue
        d = first_digit(v)
        if d == 0:
            continue
        c[d] += 1
        total += 1
    return {d: (c[d] / total if total else 0.0) for d in range(1, 10)}


def plot_benford(term_freq: Counter, title: str, out_path_base: str):
    observed = benford_distribution(term_freq.values(), drop_lt_10=False)
    observed_no1 = benford_distribution(term_freq.values(), drop_lt_10=True)
    expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}

    for suffix, dist in [('all', observed), ('ge10', observed_no1)]:
        plt.figure()
        xs = np.arange(1, 10)
        obs = np.array([dist[d] for d in xs])
        exp = np.array([expected[d] for d in xs])
        width = 0.35
        plt.bar(xs - width/2, obs, width, label='Observed')
        plt.bar(xs + width/2, exp, width, label='Benford expected')
        plt.xticks(xs)
        plt.xlabel('First digit')
        plt.ylabel('Probability')
        plt.title(f"Benford's law – {title} ({'all freqs' if suffix=='all' else 'freq≥10'})")
        plt.legend()
        plt.tight_layout()
        out_path = f"{out_path_base}_benford_{suffix}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()

# ----------- Heaps -----------

def stream_heaps(input_path: str, stopwords: set[str], sample_every: int = 10_000) -> CorpusStats:
    stemmer = PorterStemmer()
    vocab = set()
    tf = Counter()
    N = 0
    points: List[Tuple[int,int]] = []
    with smart_open(input_path) as f:
        for tok in preprocess_tokens(iter_tokens(f), stopwords, stemmer):
            N += 1
            if tok not in vocab:
                vocab.add(tok)
            tf[tok] += 1
            if N % sample_every == 0:
                points.append((N, len(vocab)))
    # ensure last point added
    if N % sample_every != 0:
        points.append((N, len(vocab)))
    return CorpusStats(term_freq=tf, heaps_points=points)


def fit_heaps(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    # Fit log V = log k + b log N
    N = np.array([p[0] for p in points], dtype=float)
    V = np.array([p[1] for p in points], dtype=float)
    N = N[N > 0]
    V = V[: len(N)]  # align lengths
    logN = np.log(N)
    logV = np.log(V)
    b, logk = np.polyfit(logN, logV, 1)  # slope b, intercept logk
    k = float(np.exp(logk))
    return k, float(b)


def plot_heaps(points: List[Tuple[int,int]], k: float, b: float, title: str, out_path: str):
    N = np.array([p[0] for p in points], dtype=float)
    V = np.array([p[1] for p in points], dtype=float)
    plt.figure()
    plt.plot(N, V, marker='.', linestyle='none', label='Observed')
    N_line = np.linspace(max(1, N.min()), N.max(), 200)
    V_fit = k * (N_line ** b)
    plt.plot(N_line, V_fit, label=f'Fit: V = {k:.2f} * N^{b:.3f}')
    plt.xlabel('N (tokens seen)')
    plt.ylabel('V (vocab size)')
    plt.title(f"Heap's law – {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ----------- Glue / CLI -----------

def load_stopwords(path: str | None) -> set[str]:
    if path is None:
        return set(DEFAULT_STOPWORDS)
    s = set()
    with smart_open(path) as f:
        for line in f:
            w = line.strip().lower()
            if not w or w.startswith('#'):
                continue
            s.add(w)
    return s


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='TTDS Lab 1 toolkit')
    p.add_argument('--collections', nargs='+', required=True, help='Paths to text files (.txt or .gz)')
    p.add_argument('--stopwords', help='Optional stopword list file (one word per line)')
    p.add_argument('--outdir', default='outputs', help='Directory for outputs (processed text + plots)')
    p.add_argument('--heaps-sample', type=int, default=10000, help='Sample interval for Heap\'s law points')
    args = p.parse_args(argv)

    data_dir = Path(__file__).resolve().parent / "assets"
    os.makedirs(args.outdir, exist_ok=True)
    stopwords = load_stopwords(data_dir / args.stopwords)

    for path in args.collections:
        base = os.path.basename(path)
        name = re.sub(r"\.(txt|gz)$", "", base)

        # 1) Preprocess & write processed file
        processed_path = os.path.join(args.outdir, f"{name}.processed.txt")
        print(f"[+] Preprocessing {path} -> {processed_path}")
        tf = write_preprocessed(data_dir / path, processed_path, stopwords)

        # 2) Zipf plots
        zipf_path = os.path.join(args.outdir, f"{name}_zipf.png")
        print(f"[+] Zipf plot -> {zipf_path}")
        plot_zipf(tf, name, zipf_path)

        # 3) Benford plots
        print(f"[+] Benford plots -> {name}_benford_*.png")
        plot_benford(tf, name, os.path.join(args.outdir, name))

        # 4) Heap's law streaming + fit
        print(f"[+] Streaming for Heap's law: {path}")
        stats = stream_heaps(data_dir / path, stopwords, sample_every=args.heaps_sample)
        k, b = fit_heaps(stats.heaps_points)
        heaps_path = os.path.join(args.outdir, f"{name}_heaps.png")
        print(f"[+] Heap's plot -> {heaps_path} | Fit: k={k:.3f}, b={b:.3f}")
        plot_heaps(stats.heaps_points, k, b, name, heaps_path)

        # 5) Save frequency table (optional but handy)
        freqs_path = os.path.join(args.outdir, f"{name}_freqs.tsv")
        with open(freqs_path, 'w', encoding='utf-8') as f:
            for term, freq in tf.most_common():
                f.write(f"{term}\t{freq}\n")
        print(f"[+] Wrote frequencies -> {freqs_path}")

    print('[✓] Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())