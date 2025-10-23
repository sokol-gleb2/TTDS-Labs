from pathlib import Path
import regex as re
from nltk.stem import PorterStemmer


def load_corpora() -> dict[str, str]:
    """Return raw text for each corpus in the assets directory."""
    data_dir = Path(__file__).resolve().parent / "assets"
    corpora = {}

    for filename in ("abstracts.wiki.txt", "pg10.txt", "quran.txt"):
        file_path = data_dir / filename
        corpora[filename] = file_path.read_text(encoding="utf-8")

    return corpora

def load_stop_words() -> set[str]:
    data_dir = Path(__file__).resolve().parent / "assets"
    with open(data_dir / "stopwordListEng.txt", "r", encoding="utf-8") as f:
        stopwords = {line.strip().casefold() for line in f if line.strip()}
    return stopwords

def preprocess(corpora):
    stopwords = load_stop_words()
    stemmer = PorterStemmer()
    
    for filename, text in corpora.items():
        # tokenisation + case folding
        tokens = [t.lower() for t in re.findall(r"\p{L}+", text, flags=re.UNICODE)]
        
        # Remove stop words + stemming
        normalized_tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]
        s = ' '.join(normalized_tokens)

        # update the dictionary entry
        corpora[filename] = s


if __name__ == "__main__":
    corpora = load_corpora()
    preprocess(corpora)
    ## print files from corpora into /processed
    for filename in ("abstracts.wiki.txt", "pg10.txt", "quran.txt"):
        file_path = Path(__file__).resolve().parent / "processed" / "processed."+filename
        text = corpora[filename]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    
