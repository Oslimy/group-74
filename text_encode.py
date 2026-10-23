# text_encode.py - Counting Bloom Filter (CBF) encoder for unstructured text,
# following the sponsor slides: bag-of-words + term-frequency counters, min-sum
# Dice similarity, with Laplace differential-privacy noise on the counters.
import re, hashlib, math
import numpy as np

_STOP = set("""a an the and or but if then else of to in on at by for with from into over under
is are was were be been being am do does did have has had this that these those it its as not no
his her he she they them we you your our their i me my mr mrs dr patient presented history year
years old male female man woman also were was had has been""".split())
_TOKEN = re.compile(r"[a-z]{3,}")
_SUFFIX = re.compile(r"\[\[[\d.]+,\s*'[a-z]+'\]\]\s*[mf]?\s*$")  # trailing [[age,'unit']] sex

def tokenize(note):
    s = str(note).replace("\\n", " ").lower()
    s = _SUFFIX.sub("", s)                      # drop the age/sex metadata suffix
    toks = [t for t in _TOKEN.findall(s) if t not in _STOP]
    return toks

def bag_of_words(note):
    bow = {}
    for t in tokenize(note):
        bow[t] = bow.get(t, 0) + 1
    return bow

class CBF:
    """Counting Bloom filter encoder with cached per-token hash positions."""
    def __init__(self, length=1000, k=10):
        self.L = length; self.k = k; self._cache = {}
    def _positions(self, tok):
        p = self._cache.get(tok)
        if p is None:
            i1 = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
            i2 = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            p = [(i1 + i * i2) % self.L for i in range(self.k)]
            self._cache[tok] = p
        return p
    def encode(self, note):
        v = np.zeros(self.L, dtype=np.float64)
        for tok, cnt in bag_of_words(note).items():
            for pos in self._positions(tok):
                v[pos] += cnt          # increment counter by term frequency
        return v

def minsum_dice(C1, C2):
    """Min-sum Dice similarity between two counting Bloom filters (matrices, row-wise)."""
    mn = np.minimum(C1, C2).sum(axis=1)
    denom = C1.sum(axis=1) + C2.sum(axis=1) + 1e-9
    return 2.0 * mn / denom

def add_dp_noise(C, epsilon, k):
    """Laplace mechanism on counting Bloom filters: scale = 2k/epsilon, clip at 0."""
    if epsilon is None:
        return C
    scale = (2.0 * k) / epsilon
    noisy = C + np.random.laplace(0.0, scale, size=C.shape)
    return np.clip(np.round(noisy), 0, None)
