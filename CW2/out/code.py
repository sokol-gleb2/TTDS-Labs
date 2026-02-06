import math
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# =========================
# Part 1: IR Evaluation
# =========================

def dcg_at_k(rels, k):
    """
    Compute DCG at cutoff k using the given equation:

    DCG_k = rel_1 + sum_{i=2}^k rel_i / log2(i)

    rels is a list of graded relevance scores in rank order.
    """
    dcg = 0.0
    limit = min(k, len(rels))
    for i in range(limit):
        rank = i + 1
        rel = rels[i]
        if rank == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(rank)
    return dcg


def compute_ir_metrics_for_list(retrieved_docs, rel_dict, total_rel):
    """
    Compute P@10, R@50, r precision, AP, nDCG@10, nDCG@20
    for one ranked list of retrieved document ids.

    retrieved_docs: list of doc ids in rank order
    rel_dict: mapping doc_id -> graded relevance (0 if missing)
    total_rel: number of relevant documents for this query (binary)
    """
    # Binary relevance for measures that treat all relevant docs as 1
    bin_rels = [1 if rel_dict.get(d, 0) > 0 else 0 for d in retrieved_docs]
    # Graded relevance for DCG based measures
    graded_rels = [rel_dict.get(d, 0) for d in retrieved_docs]

    # Precision at 10
    k = 10
    top10 = bin_rels[:k]
    p10 = sum(top10) / k if k > 0 else 0.0

    # Recall at 50
    k = 50
    top50 = bin_rels[:k]
    if total_rel > 0:
        r50 = sum(top50) / total_rel
    else:
        r50 = 0.0

    # r precision: precision at rank R, where R is number of relevant docs
    if total_rel > 0:
        r_cut = bin_rels[:total_rel]
        r_prec = sum(r_cut) / total_rel
    else:
        r_prec = 0.0

    # Average precision
    if total_rel > 0:
        num_rel_so_far = 0
        sum_prec = 0.0
        for i, rel in enumerate(bin_rels):
            if rel == 1:
                num_rel_so_far += 1
                prec_i = num_rel_so_far / (i + 1)
                sum_prec += prec_i
        ap = sum_prec / total_rel
    else:
        ap = 0.0

    # nDCG at 10 and 20
    ndcgs = {}
    for k in [10, 20]:
        dcg = dcg_at_k(graded_rels, k)
        # Ideal DCG uses all relevant docs sorted by graded relevance
        ideal_rels = sorted(
            [r for r in rel_dict.values() if r > 0],
            reverse=True
        )
        idcg = dcg_at_k(ideal_rels, k)
        if idcg > 0:
            ndcgs[k] = dcg / idcg
        else:
            ndcgs[k] = 0.0

    return p10, r50, r_prec, ap, ndcgs[10], ndcgs[20]


def run_ir_eval(
    qrels_path="qrels.csv",
    results_path="ttdssystemresults.csv",
    output_path="ir_eval.csv",
):
    """
    Load qrels and system results, compute all IR measures
    for each system and query, and write ir_eval.csv.

    Output format:
    system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20

    Includes a mean row per system, with query_number set to "mean".
    Values are rounded to three decimal places in the output file.
    """
    qrels = pd.read_csv(qrels_path)
    results = pd.read_csv(results_path)

    # Build relevance map: qrels_dict[query_id][doc_id] = relevance
    qrels_dict = {}
    for _, row in qrels.iterrows():
        q = int(row["query_id"])
        d = int(row["doc_id"])
        rel = int(row["relevance"])
        qrels_dict.setdefault(q, {})[d] = rel

    # Number of relevant documents per query (binary)
    total_rels = {
        q: sum(1 for r in docs.values() if r > 0)
        for q, docs in qrels_dict.items()
    }

    rows = []

    # Compute metrics per system and query
    for system in sorted(results["system_number"].unique()):
        sys_results = results[results["system_number"] == system]
        for query in sorted(sys_results["query_number"].unique()):
            sub = sys_results[sys_results["query_number"] == query] \
                .sort_values("rank_of_doc")
            retrieved = sub["doc_number"].tolist()

            rel_dict = qrels_dict.get(query, {})
            total_rel = total_rels.get(query, 0)

            p10, r50, r_prec, ap, ndcg10, ndcg20 = compute_ir_metrics_for_list(
                retrieved, rel_dict, total_rel
            )

            rows.append({
                "system_number": system,
                "query_number": str(query),
                "P@10": p10,
                "R@50": r50,
                "r-precision": r_prec,
                "AP": ap,
                "nDCG@10": ndcg10,
                "nDCG@20": ndcg20,
            })

        # Mean row over all ten queries for this system
        sys_df = pd.DataFrame(
            [r for r in rows
             if r["system_number"] == system and r["query_number"].isdigit()]
        )
        mean_row = {
            "system_number": system,
            "query_number": "mean",
            "P@10": sys_df["P@10"].mean(),
            "R@50": sys_df["R@50"].mean(),
            "r-precision": sys_df["r-precision"].mean(),
            "AP": sys_df["AP"].mean(),
            "nDCG@10": sys_df["nDCG@10"].mean(),
            "nDCG@20": sys_df["nDCG@20"].mean(),
        }
        rows.append(mean_row)

    ir_eval = pd.DataFrame(rows)

    # Order columns exactly as required
    ir_eval = ir_eval[
        [
            "system_number",
            "query_number",
            "P@10",
            "R@50",
            "r-precision",
            "AP",
            "nDCG@10",
            "nDCG@20",
        ]
    ]

    # Round to three decimal places in the CSV output
    ir_eval_rounded = ir_eval.copy()
    for col in ["P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"]:
        ir_eval_rounded[col] = ir_eval_rounded[col].map(
            lambda x: f"{x:.3f}"
        )

    ir_eval_rounded.to_csv(output_path, index=False)
    print(f"Saved IR evaluation to {output_path}")


# =========================
# Part 2: Text Analysis
# =========================

def preprocess_and_vectorize_text(tsv_path="bible_and_quran.tsv"):
    """
    Read the verse file and create a document term matrix.

    The input file is a TSV with:
        corpus_name <tab> verse text

    We treat each line as one document. Preprocessing uses
    CountVectorizer with:
        - lowercasing
    tokenisation on words
        - English stopword removal
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["corpus", "text"],
    )

    texts = df["text"].astype(str).tolist()
    labels = df["corpus"].astype(str).to_numpy()

    # Tokenisation and stopword removal
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    print("Text matrix shape:", X.shape)
    print("Corpora counts:")
    print(df["corpus"].value_counts())

    return X, labels, feature_names


def write_ranked_scores(tokens, scores, out_path, top_n=None):
    """
    Save a ranked list token,score to a CSV file.

    tokens: list or array of token strings
    scores: numeric scores for each token
    out_path: output CSV path
    top_n: if given, limit to top N tokens
    """
    # Fix the output path so that files are written
    # relative to the location of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, out_path)

    # Combine and sort descending by score
    data = list(zip(tokens, scores))
    # Filter out tokens whose score is zero for cleaner output
    data = [(t, s) for t, s in data if s > 0]
    data.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        data = data[:top_n]

    # Build dataframe and round scores to three decimals
    df = pd.DataFrame(data, columns=["token", "score"])
    df["score"] = df["score"].map(lambda x: f"{x:.3f}")
    df.to_csv(out_path, index=False)
    print(f"Wrote ranked scores to {out_path}")


def compute_token_scores_per_corpus(X, labels, feature_names):
    """
    For each corpus (Quran, OT, NT) compute:

        Mutual Information scores for all tokens (one corpus versus the rest)
        Chi square scores for all tokens (one corpus versus the rest)

    Output files:
        mi_<corpus>.csv
        chi2_<corpus>.csv

    Each file is in the format:
        token,score

    Implementation note:
    - We treat features as binary (present/absent in a document).
    - For each corpus, we build 2x2 contingency tables for every token:
          term present   term absent
      y=1       A             C
      y=0       B             D
    and compute MI and χ² from those counts.
    """
    from scipy import sparse

    # Convert counts to binary presence/absence
    X_bin = X.copy().tocsr()
    X_bin.data = np.ones_like(X_bin.data)

    n_docs, n_terms = X_bin.shape
    corpora = sorted(set(labels))

    for corpus in corpora:
        print(f"Computing scores for corpus {corpus}")

        # Binary class labels: 1 if this corpus, 0 otherwise
        y = (labels == corpus).astype(int)
        y_pos = y
        y_neg = 1 - y

        n_pos = int(y_pos.sum())
        n_neg = int(y_neg.sum())
        N = n_pos + n_neg

        # Represent y as 1xN sparse rows so we can do y * X efficiently
        y_pos_row = sparse.csr_matrix(y_pos.reshape(1, -1))
        y_neg_row = sparse.csr_matrix(y_neg.reshape(1, -1))

        # A: docs in class (y=1) where term is present
        # B: docs not in class (y=0) where term is present
        A = y_pos_row.dot(X_bin).toarray().ravel().astype(float)
        B = y_neg_row.dot(X_bin).toarray().ravel().astype(float)

        # C: docs in class where term is absent
        # D: docs not in class where term is absent
        C = n_pos - A
        D = n_neg - B

        # ----- Mutual Information -----
        # Probabilities
        PwC   = A / N
        PwNc  = B / N
        PnWc  = C / N
        PnWNc = D / N

        Pw  = (A + B) / N
        PnW = (C + D) / N
        Pc  = n_pos / N
        PNc = n_neg / N

        mi_scores = np.zeros(n_terms, dtype=float)

        # Add the four terms of MI, skipping zero joint probabilities
        mask = PwC > 0
        mi_scores[mask] += PwC[mask] * np.log2(PwC[mask] / (Pw[mask] * Pc))

        mask = PwNc > 0
        mi_scores[mask] += PwNc[mask] * np.log2(PwNc[mask] / (Pw[mask] * PNc))

        mask = PnWc > 0
        mi_scores[mask] += PnWc[mask] * np.log2(PnWc[mask] / (PnW[mask] * Pc))

        mask = PnWNc > 0
        mi_scores[mask] += PnWNc[mask] * np.log2(PnWNc[mask] / (PnW[mask] * PNc))

        # ----- Chi-square -----
        # χ² = N * (AD - BC)^2 / ((A+B)(C+D)(A+C)(B+D))
        num = (A * D - B * C) ** 2 * N
        den = (A + B) * (C + D) * (A + C) * (B + D)

        chi2_scores = np.zeros(n_terms, dtype=float)
        mask = den > 0
        chi2_scores[mask] = num[mask] / den[mask]

        # Output file names
        mi_path = f"mi_{corpus}.csv"
        chi2_path = f"chi2_{corpus}.csv"

        write_ranked_scores(feature_names, mi_scores, mi_path)
        write_ranked_scores(feature_names, chi2_scores, chi2_path)


def get_top_tokens_for_topic(topic_word_matrix, feature_names, topic_index, top_n=10):
    """
    Given the topic word matrix from LDA and a topic index,
    return the top N tokens with the highest probability
    in that topic.

    topic_word_matrix: array [n_topics, n_features]
    feature_names: array of token strings
    topic_index: integer index of the topic
    """
    topic_vector = topic_word_matrix[topic_index]
    # Larger values correspond to higher probability in that topic
    top_indices = np.argsort(topic_vector)[::-1][:top_n]
    top_tokens = [feature_names[i] for i in top_indices]
    top_values = topic_vector[top_indices]
    return list(zip(top_tokens, top_values))


def run_lda_and_corpus_topics(
    X,
    labels,
    feature_names,
    n_topics=20,
    output_path="topics_by_corpus.txt",
):
    """
    Run LDA on all verses together (Quran, OT, NT) and then:

      For each corpus, compute average topic scores by averaging
        document topic probabilities over all documents in that corpus.
      Identify the topic with the highest average score for each corpus.
      For each selected topic, list the top 10 tokens.

    The summary is written to topics_by_corpus.txt.
    """
    print("Running LDA with", n_topics, "topics")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=0,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(X)        # shape: [n_docs, n_topics]
    topic_word = lda.components_           # shape: [n_topics, n_features]

    corpora = sorted(set(labels))

    lines = []
    for corpus in corpora:
        mask = (labels == corpus)
        corpus_doc_topic = doc_topic[mask]

        # Average topic distribution over documents in this corpus
        avg_topic_scores = corpus_doc_topic.mean(axis=0)
        best_topic = int(np.argmax(avg_topic_scores))
        best_score = float(avg_topic_scores[best_topic])

        # Top 10 tokens for this topic
        top_tokens_with_values = get_top_tokens_for_topic(
            topic_word, feature_names, best_topic, top_n=10
        )
        # Only keep token strings for human readable output
        top_tokens = [t for t, _ in top_tokens_with_values]

        lines.append(f"Corpus: {corpus}")
        lines.append(f"  Best topic index: {best_topic}")
        lines.append(f"  Average topic score: {best_score:.3f}")
        lines.append("  Top 10 tokens for this topic:")
        for tok in top_tokens:
            lines.append(f"    {tok}")
        lines.append("")  # blank line between corpora sections

    # Write out relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_full = os.path.join(script_dir, output_path)
    with open(out_full, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

    print(f"Wrote topic summary per corpus to {out_full}")


# =========================
# Part 3: Text Classification (Sentiment Analysis)
# =========================

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def load_tweet_data(txt_path):
    """
    Load the tweet sentiment dataset.

    The file is a tab separated file with a header:
        id    sentiment    tweet

    Returns a DataFrame with columns: id, sentiment, tweet.
    """
    df = pd.read_csv(
        txt_path,
        sep="\t",
        header=0,
        names=["id", "sentiment", "tweet"],
        quoting=3,  # avoid treating quotes as special
        encoding="utf8",
    )
    # Drop rows with missing text or label, if any
    df = df.dropna(subset=["sentiment", "tweet"])
    return df


def evaluate_classifier(clf, X, y, system_name, split_name):
    """
    Compute precision, recall and F1 for each class and macro-average,
    print them, and return a dict compatible with the required CSV format.
    """
    y_pred = clf.predict(X)

    # We fix the label order so mapping to pos/neg/neu is explicit
    label_order = ["positive", "negative", "neutral"]

    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=label_order, zero_division=0
    )

    # Build per-class metrics dict
    stats = {
        label: {
            "p": float(p),
            "r": float(r),
            "f": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(label_order, prec, rec, f1, support)
    }

    # Macro averages over the three classes
    macro_p = float(prec.mean())
    macro_r = float(rec.mean())
    macro_f = float(f1.mean())

    # Print for human inspection (as before)
    print(f"\n=== Evaluation on {split_name} set ({system_name}) ===")
    for label in label_order:
        s = stats[label]
        print(
            f"Class '{label}': "
            f"precision={s['p']:.3f}, recall={s['r']:.3f}, f1={s['f']:.3f}, "
            f"support={s['support']}"
        )
    print(
        f"Macro average: precision={macro_p:.3f}, "
        f"recall={macro_r:.3f}, f1={macro_f:.3f}"
    )

    print("\nDetailed classification report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=label_order,
            zero_division=0,
            digits=3,
        )
    )

    # Map to the required header format
    row = {
        "system": system_name,
        "split": split_name,
        "p-pos": stats["positive"]["p"],
        "r-pos": stats["positive"]["r"],
        "f-pos": stats["positive"]["f"],
        "p-neg": stats["negative"]["p"],
        "r-neg": stats["negative"]["r"],
        "f-neg": stats["negative"]["f"],
        "p-neu": stats["neutral"]["p"],
        "r-neu": stats["neutral"]["r"],
        "f-neu": stats["neutral"]["f"],
        "p-macro": macro_p,
        "r-macro": macro_r,
        "f-macro": macro_f,
    }
    return row


def run_sentiment_baseline(X_train_texts, y_train,
                           X_dev_texts, y_dev,
                           X_test_texts, y_test):
    """
    Baseline sentiment classifier using:
        - unigram bag-of-words with stopword removal
        - Linear SVM (LinearSVC) with C=1000

    Trains on the provided training split and evaluates on
    train, dev, and test splits.
    """
    print("\nRunning baseline sentiment classifier...")

    # Unigram BOW with English stopwords removed
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_dev = vectorizer.transform(X_dev_texts)
    X_test = vectorizer.transform(X_test_texts)

    clf = LinearSVC(C=1000, random_state=0, max_iter=10000)
    clf.fit(X_train, y_train)

    rows = []
    rows.append(
        evaluate_classifier(
            clf, X_train, y_train,
            system_name="baseline", split_name="train"
        )
    )
    rows.append(
        evaluate_classifier(
            clf, X_dev, y_dev,
            system_name="baseline", split_name="dev"
        )
    )
    rows.append(
        evaluate_classifier(
            clf, X_test, y_test,
            system_name="baseline", split_name="test"
        )
    )

    return rows


def run_sentiment_improved(X_train_texts, y_train,
                           X_dev_texts, y_dev,
                           X_test_texts, y_test):
    """
    Improved sentiment classifier that builds on the baseline but
    modifies both features and model to improve development performance.

    Changes relative to the baseline:
      - TF-IDF features with unigrams and bigrams
      - Keep stopwords to capture negations ("not", "no")
      - Linear SVM with smaller C to reduce overfitting

    Trains on the provided training split and evaluates on
    train, dev, and test splits.
    """
    print("\nRunning improved sentiment classifier...")

    # TF-IDF with unigrams and bigrams; keep stopwords
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train = tfidf.fit_transform(X_train_texts)
    X_dev = tfidf.transform(X_dev_texts)
    X_test = tfidf.transform(X_test_texts)

    clf = LinearSVC(C=1.0, random_state=0, max_iter=20000)
    clf.fit(X_train, y_train)

    rows = []
    rows.append(
        evaluate_classifier(
            clf, X_train, y_train,
            system_name="improved", split_name="train"
        )
    )
    rows.append(
        evaluate_classifier(
            clf, X_dev, y_dev,
            system_name="improved", split_name="dev"
        )
    )
    rows.append(
        evaluate_classifier(
            clf, X_test, y_test,
            system_name="improved", split_name="test"
        )
    )

    return rows


# =========================
# Main driver
# =========================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Part 1: IR evaluation
    run_ir_eval(
        qrels_path=os.path.join(script_dir, "qrels.csv"),
        results_path=os.path.join(script_dir, "ttdssystemresults.csv"),
        output_path=os.path.join(script_dir, "ir_eval.csv"),
    )

    # Part 2: Text analysis
    X, labels, feature_names = preprocess_and_vectorize_text(
        tsv_path=os.path.join(script_dir, "bible_and_quran.tsv")
    )
    compute_token_scores_per_corpus(X, labels, feature_names)
    run_lda_and_corpus_topics(
        X,
        labels,
        feature_names,
        n_topics=20,
        output_path=os.path.join(script_dir, "topics_by_corpus.txt"),
    )

        # Part 3: Sentiment classification

    # Load TRAIN data
    train_df = load_tweet_data(os.path.join(script_dir, "train.txt"))
    X_all = train_df["tweet"].astype(str).to_numpy()
    y_all = train_df["sentiment"].astype(str).to_numpy()

    # Split TRAIN into train/dev
    X_train_texts, X_dev_texts, y_train, y_dev = train_test_split(
        X_all,
        y_all,
        test_size=0.1,
        random_state=0,
        stratify=y_all,
    )

    # Load TEST data
    test_df = load_tweet_data(os.path.join(script_dir, "ttds_2025_cw2_test.txt"))
    X_test_texts = test_df["tweet"].astype(str).to_numpy()
    y_test = test_df["sentiment"].astype(str).to_numpy()

    # Run baseline and improved classifiers on the same splits
    baseline_rows = run_sentiment_baseline(
        X_train_texts, y_train,
        X_dev_texts, y_dev,
        X_test_texts, y_test,
    )
    improved_rows = run_sentiment_improved(
        X_train_texts, y_train,
        X_dev_texts, y_dev,
        X_test_texts, y_test,
    )

    all_rows = baseline_rows + improved_rows

    # Make sure the columns are in the exact order requested
    cols = [
        "system", "split",
        "p-pos", "r-pos", "f-pos",
        "p-neg", "r-neg", "f-neg",
        "p-neu", "r-neu", "f-neu",
        "p-macro", "r-macro", "f-macro",
    ]
    df_out = pd.DataFrame(all_rows)[cols]

    out_path = os.path.join(script_dir, "classification.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote sentiment results to {out_path}")


if __name__ == "__main__":
    main()