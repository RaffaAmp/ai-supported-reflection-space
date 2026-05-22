# bm25_search.py
"""
BM25-based search for the Windpark Lindenberg knowledge base.
Drop-in replacement for the old improved_search() function.
"""

import re
from rank_bm25 import BM25Okapi


# German stopwords - common words to ignore during search
GERMAN_STOPWORDS = {
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen',
    'einem', 'einer', 'eines', 'und', 'oder', 'aber', 'ist', 'sind',
    'war', 'waren', 'wird', 'werden', 'wie', 'was', 'wer', 'wo', 'wann',
    'warum', 'wieso', 'mit', 'von', 'zu', 'bei', 'auf', 'fuer', 'fur',
    'in', 'an', 'am', 'im', 'aus', 'nach', 'vor', 'ueber', 'uber',
    'unter', 'durch', 'gegen', 'ohne', 'um', 'ich', 'du', 'er', 'sie',
    'es', 'wir', 'ihr', 'man', 'sich', 'mich', 'dich', 'mir', 'dir',
    'nicht', 'auch', 'noch', 'nur', 'schon', 'sehr', 'mehr', 'so',
    'dass', 'weil', 'wenn', 'als', 'haben', 'hat', 'hatte',
    'sein', 'seine', 'seinen', 'kann', 'koennen', 'konnen', 'soll',
    'sollen', 'will', 'wollen', 'muss', 'muessen', 'mussen',
}


def normalize_text(text):
    """Normalize German text for consistent matching."""
    text = text.lower()
    text = text.replace('ß', 'ss')
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    return text


def tokenize(text):
    """Split text into searchable tokens, removing stopwords."""
    text = normalize_text(text)
    words = re.findall(r'\w+', text)
    return [w for w in words if len(w) > 2 and w not in GERMAN_STOPWORDS]


# Topic keyword boosting (kept from the original search)
TOPIC_KEYWORDS = {
    'umwelt':   ['umwelt', 'natur', 'oekologie', 'okologie', 'lebensraum'],
    'laerm':    ['laerm', 'larm', 'schall', 'geraeusch', 'gerausch', 'dezibel'],
    'energie':  ['energie', 'strom', 'leistung', 'kwh', 'mw', 'gwh'],
    'bau':      ['bau', 'errichtung', 'konstruktion', 'montage'],
    'kosten':   ['kosten', 'preis', 'finanzierung', 'investition'],
    'klima':    ['klima', 'klimastrategie', 'klimawandel', 'klimaschutz'],
    'emission': ['emission', 'treibhausgas', 'co2', 'kohlendioxid'],
    'netto':    ['netto', 'null', 'klimaneutral', 'klimaneutralitaet'],
    'ziel':     ['2050', 'ziel', 'strategie', 'bundesrat'],
    'schweiz':  ['schweiz', 'schweizerisch', 'eidgenossenschaft', 'bund'],
}


class BM25Searcher:
    """BM25-based document searcher with German language support."""

    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [tokenize(doc["content"]) for doc in documents]
        # Handle edge case: empty docs would crash BM25
        self.tokenized_docs = [
            tokens if tokens else ["__empty__"]
            for tokens in self.tokenized_docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, max_results=3, min_score=0.5):
        """Search documents and return top matches above min_score threshold."""
        if not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Apply topic keyword boost and exact phrase bonus
        query_norm = normalize_text(query)
        for i, doc in enumerate(self.documents):
            content_norm = normalize_text(doc["content"])

            # Topic keyword boost
            for topic, related_words in TOPIC_KEYWORDS.items():
                query_has_topic = any(kw in query_norm for kw in related_words)
                doc_has_topic = any(kw in content_norm for kw in related_words)
                if query_has_topic and doc_has_topic:
                    scores[i] += 2.0

            # Exact phrase bonus
            if len(query_norm) > 5 and query_norm in content_norm:
                scores[i] += 5.0

        # Filter by minimum score and return top results
        scored_docs = [
            (scores[i], self.documents[i])
            for i in range(len(self.documents))
            if scores[i] >= min_score
        ]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]


# Module-level cache so we don't rebuild the BM25 index on every search
_searcher_cache = None
_cached_doc_count = 0


def improved_search(query, documents, max_results=3):
    """
    Drop-in replacement for the old improved_search().
    Uses BM25 with topic boosting and German normalization.
    """
    global _searcher_cache, _cached_doc_count

    if not documents:
        return []

    # Rebuild index only if document count changed (or first call)
    if _searcher_cache is None or _cached_doc_count != len(documents):
        print(f"🔧 Building BM25 index for {len(documents)} documents...")
        _searcher_cache = BM25Searcher(documents)
        _cached_doc_count = len(documents)
        print(f"✅ BM25 index ready")

    return _searcher_cache.search(query, max_results=max_results)
