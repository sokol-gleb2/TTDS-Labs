#!/usr/bin/env python3
"""
TTDS Lab 3 ranked retrieval using TF IDF

This script assumes you have already built the XML index with Lab 2 code
and have these files in some directory (default "data"):

    index.json       term -> {doc_id: [positions]}
    df.json          term -> document frequency
    doc_lengths.json doc_id -> length in tokens

This script:
  - loads that index
  - runs ranked retrieval for a set of free text queries
  - writes results to "tfidf.results" as:
        qid,docid,score
    sorted first by qid increasing, then by score descending.
"""

from __future__ import annotations
import argparse
import io
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Iterable, Tuple, Optional, Set

# ----------------------------------------------------------------------
# Preprocessing (same tokenizer and default stopwords as Lab 2)
# ----------------------------------------------------------------------

DEFAULT_STOPWORDS = {
    "a","an","the","and","or","not","is","are","was","were","be","been","being",
    "of","to","in","on","for","with","as","by","at","from","that","this","these","those",
    "it","its","but","if","then","than","so","such","too","very","into","about","over","under",
    "no","nor","can","could","would","should","will","shall","may","might","must","do","does","did",
    "have","has","had","having","you","your","yours","he","she","they","them","his","her","their","we","our",
    "i","me","my","mine","him","hers","theirs","us","ours"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

class OptionalPorter:
    """
    Uses NLTK Porter stemmer if available.
    If import fails, stemming becomes a no op.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stemmer = None
        if enabled:
            try:
                from nltk.stem import PorterStemmer  # type: ignore
                self._stemmer = PorterStemmer()
            except Exception:
                self.enabled = False

    def stem(self, token: str) -> str:
        if not self.enabled or not self._stemmer:
            return token
        return self._stemmer.stem(token)

def preprocess_query(
    text: str,
    *,
    use_stops: bool = True,
    use_stem: bool = True,
    stopwords: Optional[Set[str]] = None
) -> List[str]:
    stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    stemmer = OptionalPorter(enabled=use_stem)
    tokens = tokenize(text)
    out: List[str] = []
    for tok in tokens:
        if use_stops and tok in stopwords:
            continue
        if use_stem:
            tok = stemmer.stem(tok)
        out.append(tok)
    return out

# ----------------------------------------------------------------------
# Index loading
# ----------------------------------------------------------------------

class Index:
    def __init__(
        self,
        postings: Dict[str, Dict[str, List[int]]],
        df: Dict[str, int],
        doc_lengths: Dict[str, int],
    ):
        self.postings = postings
        self.df = df
        self.doc_lengths = doc_lengths
        self.N = len(doc_lengths)

    @staticmethod
    def load(base_dir: str) -> "Index":
        with open(os.path.join(base_dir, "index.json"), "r", encoding="utf-8") as f:
            postings = json.load(f)
        with open(os.path.join(base_dir, "doc_lengths.json"), "r", encoding="utf-8") as f:
            doc_lengths = json.load(f)
        if os.path.exists(os.path.join(base_dir, "df.json")):
            with open(os.path.join(base_dir, "df.json"), "r", encoding="utf-8") as f:
                df = json.load(f)
        else:
            # fallback if df was not separately stored
            df = {t: len(pl) for t, pl in postings.items()}

        # normalise types
        postings = {
            term: {doc: list(map(int, pos_list)) for doc, pos_list in pl.items()}
            for term, pl in postings.items()
        }
        doc_lengths = {doc: int(n) for doc, n in doc_lengths.items()}
        df = {term: int(v) for term, v in df.items()}

        return Index(postings=postings, df=df, doc_lengths=doc_lengths)

# ----------------------------------------------------------------------
# TF IDF scoring
# ----------------------------------------------------------------------

def tf_raw_to_weight(freq: int) -> float:
    """
    Document or query term frequency to TF weight.
    Uses 1 + log10(freq) for freq > 0, else 0.
    """
    if freq <= 0:
        return 0.0
    return 1.0 + math.log10(freq)

def idf(term: str, index: Index) -> float:
    """
    Smoothed IDF:
        idf(t) = log10((N + 1) / (df(t) + 1)) + 1
    """
    N = index.N
    df_t = index.df.get(term, 0)
    return math.log10((N + 1.0) / (df_t + 1.0)) + 1.0

def score_query_tfidf(
    query: str,
    index: Index,
    *,
    use_stops: bool = True,
    use_stem: bool = True
) -> Dict[str, float]:
    """
    Compute TF IDF scores for a single free text query.

    Scoring formula:

      tf_doc = 1 + log10(freq_doc)
      tf_q   = 1 + log10(freq_query)
      idf_t  = log10((N + 1) / (df_t + 1)) + 1

      score(d, q) = sum over t in q intersect d of
                    tf_doc(t, d) * tf_q(t, q) * idf_t^2

    Returns a mapping: doc_id -> score
    """
    # preprocess query
    q_terms = preprocess_query(query, use_stops=use_stops, use_stem=use_stem)
    if not q_terms:
        return {}

    # term frequency in query
    q_tf_raw = Counter(q_terms)

    # pre compute query term weights
    q_tf: Dict[str, float] = {}
    q_idf: Dict[str, float] = {}
    for t, freq in q_tf_raw.items():
        q_tf[t] = tf_raw_to_weight(freq)
        q_idf[t] = idf(t, index)

    scores: Dict[str, float] = {}

    # for each query term, walk its postings
    for t in q_terms:
        if t not in index.postings:
            continue
        posting = index.postings[t]
        tf_q = q_tf[t]
        idf_t = q_idf[t]
        weight_q = tf_q * (idf_t * idf_t)  # tf_q * idf^2

        for doc_id, positions in posting.items():
            tf_d = tf_raw_to_weight(len(positions))
            contrib = tf_d * weight_q
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib

    return scores

# ----------------------------------------------------------------------
# Queries and running
# ----------------------------------------------------------------------

DEFAULT_QUERIES = {
    1: "income tax reduction",
    2: "peace in the Middle East",
    3: "unemployment rate in UK",
    4: "industry in scotland",
    5: "the industries of computers",
    6: "Microsoft Windows",
    7: "stock market in Japan",
    8: "the education with computers",
    9: "health industry",
    10: "campaigns of political parties",
}

def read_queries_from_file(path: str) -> Dict[int, str]:
    """
    Read queries from a file in format:

        1 income tax reduction
        2 peace in the Middle East
        ...

    Returns {qid: query_text}
    """
    queries: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                qid = int(parts[0])
            except ValueError:
                continue
            queries[qid] = parts[1].strip()
    return queries

def run_ranked_retrieval(
    index_dir: str,
    out_path: str,
    queries_file: Optional[str],
    use_stops: bool,
    use_stem: bool,
) -> None:
    index = Index.load(index_dir)

    if queries_file:
        queries = read_queries_from_file(queries_file)
    else:
        queries = DEFAULT_QUERIES

    # ensure processing in query id order
    qids = sorted(queries.keys())

    lines_out: List[str] = []

    for qid in qids:
        query_text = queries[qid]
        scores = score_query_tfidf(
            query_text,
            index,
            use_stops=use_stops,
            use_stem=use_stem,
        )
        # sort docs by score descending, then doc_id for stable order
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        for doc_id, score in ranked:
            lines_out.append(f"{qid},{doc_id},{score:.4f}")

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")

    print(f"Wrote {len(lines_out)} result lines to {out_path}")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TTDS Lab 3 ranked retrieval with TF IDF"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data",
        help="Directory that contains index.json, df.json, doc_lengths.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tfidf.results",
        help="Output file for ranked results",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Optional queries file; if omitted, built in ten queries are used",
    )
    parser.add_argument(
        "--no-stops",
        action="store_true",
        help="Disable stopword removal for queries",
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Disable stemming for queries",
    )

    args = parser.parse_args()

    run_ranked_retrieval(
        index_dir=args.index,
        out_path=args.out,
        queries_file=args.queries_file,
        use_stops=not args.no_stops,
        use_stem=not args.no_stem,
    )

if __name__ == "__main__":
    main()