"""
TTDS Lab 5
Pseudo relevance feedback for query expansion

This script:

  1. Loads the inverted index built in Lab 2 and used in Lab 3.
  2. Loads the ranked results from Lab 3 (results file).
  3. For each query, takes the top n_d documents as pseudo relevant.
  4. Collects all terms from these documents and scores them with:

       score(t) = tf(t) * log(N / df(t))

     where:
       tf(t)  = total frequency of term t across the chosen documents
       df(t)  = number of documents in the collection containing t
       N      = total number of documents in the collection

  5. Sorts terms by this score and picks the top n_t as expansion terms.
  6. Writes an expanded query file named: Qm.n_d.n_t.txt

Usage:

  python ttds_lab5_prf.py \
      --index data \
      --results results.ranked.txt \
      --nd 1 \
      --nt 5

This will create Qm.1.5.txt in the current directory.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

# -----------------------------------------------------------
# Preprocessing (same style as Lab 3)
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# Index loading
# -----------------------------------------------------------

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
            df = {t: len(pl) for t, pl in postings.items()}

        # normalise types
        postings = {
            term: {doc: list(map(int, pos_list)) for doc, pos_list in pl.items()}
            for term, pl in postings.items()
        }
        doc_lengths = {doc: int(n) for doc, n in doc_lengths.items()}
        df = {term: int(v) for term, v in df.items()}

        return Index(postings=postings, df=df, doc_lengths=doc_lengths)

# -----------------------------------------------------------
# Lab 3 queries (default)
# -----------------------------------------------------------

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
    Reads queries from a file with lines:

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

# -----------------------------------------------------------
# Read Lab 3 ranked results
# -----------------------------------------------------------

def read_ranked_results(path: str) -> Dict[int, List[str]]:
    """
    Reads ranked results from Lab 3 in format:

        qid,docid,score

    Returns mapping: qid -> [docid1, docid2, ...] in ranked order.
    """
    results: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                qid = int(parts[0])
                docid = parts[1]
                score = float(parts[2])
            except ValueError:
                continue
            results[qid].append((docid, score))

    # sort each list by score descending, then docid
    ranked: Dict[int, List[str]] = {}
    for qid, lst in results.items():
        lst_sorted = sorted(lst, key=lambda x: (-x[1], x[0]))
        ranked[qid] = [docid for docid, _ in lst_sorted]
    return ranked

# -----------------------------------------------------------
# Build document term frequencies from postings
# -----------------------------------------------------------

def build_doc_term_freqs(index: Index) -> Dict[str, Dict[str, int]]:
    """
    Builds a mapping: doc_id -> {term: freq_in_doc}
    from the inverted index postings.
    """
    doc_tf: Dict[str, Dict[str, int]] = defaultdict(dict)
    for term, posting in index.postings.items():
        for doc_id, positions in posting.items():
            doc_tf[doc_id][term] = len(positions)
    return doc_tf

# -----------------------------------------------------------
# Pseudo relevance feedback core
# -----------------------------------------------------------

def prf_expand_for_query(
    qid: int,
    query_text: str,
    top_docs: List[str],
    index: Index,
    doc_tf: Dict[str, Dict[str, int]],
    n_d: int,
    n_t: int,
    use_stops: bool = True,
    use_stem: bool = True,
) -> Tuple[int, str, List[str]]:
    """
    Performs pseudo relevance feedback expansion for one query.

    Returns:
      (qid, query_tokens_string, expansion_terms_list)
    """
    # preprocess original query
    q_tokens = preprocess_query(
        query_text,
        use_stops=use_stops,
        use_stem=use_stem,
    )

    # take top n_d documents (or fewer if not enough)
    chosen_docs = top_docs[:n_d]

    # aggregate tf across chosen docs
    tf_total: Counter[str] = Counter()
    for doc_id in chosen_docs:
        terms_in_doc = doc_tf.get(doc_id, {})
        for term, freq in terms_in_doc.items():
            tf_total[term] += freq

    # compute tf * log(N / df) for each term
    N = index.N
    scores: Dict[str, float] = {}
    for term, tf in tf_total.items():
        df_t = index.df.get(term, 0)
        if df_t <= 0:
            continue
        # lab formula: tf * log(N / df)
        weight = tf * math.log(N / df_t)
        scores[term] = weight

    # sort terms by score descending, then by term
    ranked_terms = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # pick top n_t terms
    expansion_terms = [t for t, _ in ranked_terms[:n_t]]

    # Construct the preprocessed query string
    query_str = " ".join(q_tokens)

    return qid, query_str, expansion_terms

# -----------------------------------------------------------
# Driver
# -----------------------------------------------------------

def run_prf(
    index_dir: str,
    results_path: str,
    n_d: int,
    n_t: int,
    queries_file: Optional[str],
    use_stops: bool,
    use_stem: bool,
) -> None:
    index = Index.load(index_dir)
    ranked = read_ranked_results(results_path)

    if queries_file:
        queries = read_queries_from_file(queries_file)
    else:
        queries = DEFAULT_QUERIES

    doc_tf = build_doc_term_freqs(index)

    # ensure deterministic order of queries
    qids = sorted(queries.keys())

    lines_out: List[str] = []

    for qid in qids:
        if qid not in ranked:
            continue
        query_text = queries[qid]
        top_docs = ranked[qid]

        qid_out, q_str, expansion_terms = prf_expand_for_query(
            qid=qid,
            query_text=query_text,
            top_docs=top_docs,
            index=index,
            doc_tf=doc_tf,
            n_d=n_d,
            n_t=n_t,
            use_stops=use_stops,
            use_stem=use_stem,
        )
        # format: qid preprocessed_query + term1 term2 ...
        line = f"{qid_out} {q_str} + " + " ".join(expansion_terms)
        lines_out.append(line)

    out_name = f"Qm.{n_d}.{n_t}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")

    print(f"Wrote {len(lines_out)} expanded queries to {out_name}")

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TTDS Lab 5 pseudo relevance feedback based query expansion"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data",
        help="Directory that contains index.json, df.json, doc_lengths.json",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.ranked.txt",
        help="Ranked results file from Lab 3 (qid,docid,score)",
    )
    parser.add_argument(
        "--nd",
        type=int,
        default=1,
        help="Number of top documents to use per query",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=5,
        help="Number of expansion terms to select per query",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Optional queries file in format: qid text",
    )
    parser.add_argument(
        "--no-stops",
        action="store_true",
        help="Disable stopword removal for query preprocessing",
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Disable stemming for query preprocessing",
    )

    args = parser.parse_args()

    run_prf(
        index_dir=args.index,
        results_path=args.results,
        n_d=args.nd,
        n_t=args.nt,
        queries_file=args.queries_file,
        use_stops=not args.no_stops,
        use_stem=not args.no_stem,
    )

if __name__ == "__main__":
    main()