# run_tests.py - Group 74 PPRL verification test cases TC01-TC15.
# Mirrors the Security Test Plan / Test Cases documents (D7).
# Run from the project folder:   python3 run_tests.py      (about 1-2 minutes)
import warnings, random, sys, time; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from PPRL import Link
from ml_linkage import prepare, run_cv, pair_features, bf_to_np, _quiet
from text_encode import tokenize, bag_of_words, CBF, minsum_dice, add_dp_noise

L, K, Q = 1000, 10, 2
LR = {"LR": lambda: LogisticRegression(max_iter=2000, class_weight="balanced")}
results = []

def test(tc_id, name, owner):
    def wrap(fn):
        try:
            ok, evidence = fn()
        except Exception as e:
            ok, evidence = False, f"ERROR: {type(e).__name__}: {e}"
        results.append((tc_id, name, owner, ok, evidence))
        print(f"{tc_id}  {'PASS' if ok else 'FAIL'}  {name:<34} {evidence}", flush=True)
        return fn
    return wrap

random.seed(0); np.random.seed(0)
link = Link(L, K, Q, 0.8, [1, 2, 3, 4], [2], 0, 7)
ALICE = "Datasets/Alice_numrec_100_corr_50.csv"
db = bi = bfs = None

def bf_of(s):
    return link.bf.set_to_bloom_filter(link.bf.convert_str_val_to_set(s))

print("Group 74 PPRL - verification tests\n" + "-" * 78)
t0 = time.time()

@test("TC01", "Dataset loads", "PPRL")
def _():
    global db
    db = _quiet(link.read_database, ALICE)
    return len(db) == 100, f"{len(db)} records loaded"

@test("TC02", "Blocking builds an index", "PPRL")
def _():
    global bi
    bi = _quiet(link.build_BI, db)
    return 0 < len(bi) < len(db), f"{len(bi)} blocks from {len(db)} records"

@test("TC03", "Bloom filter length (SR-01)", "PPRL")
def _():
    global bfs
    bfs, _ = _quiet(link.data_encode, db)
    lens = {len(b) for b in bfs.values()}
    return lens == {L}, f"filter length = {sorted(lens)}"

@test("TC04", "Self-similarity = 1.0", "PPRL")
def _():
    b = next(iter(bfs.values()))
    s = link.bf.calc_bf_sim(b, b)
    return abs(s - 1.0) < 1e-12, f"{s:.1f}"

@test("TC05", "Similar vs dissimilar names", "PPRL")
def _():
    a, b, c = bf_of("smith"), bf_of("smyth"), bf_of("johnson")
    s1, s2 = link.bf.calc_bf_sim(a, b), link.bf.calc_bf_sim(a, c)
    return s1 > s2, f"{s1:.2f} > {s2:.2f}"

@test("TC06", "DP noise perturbs filters (SR-03)", "PPRL")
def _():
    noisy = link.add_DP_noise(bfs)
    changed = sum(noisy[r] != bfs[r] for r in bfs)
    return changed == len(bfs), f"{changed}/{len(bfs)} filters changed"

@test("TC07", "Epsilon controls noise (SR-03)", "PPRL")
def _():
    def flips(eps):
        lk = Link(L, K, Q, 0.8, [1, 2, 3, 4], [2], 0, eps)
        r = next(iter(bfs)); n = lk.add_DP_noise({r: bfs[r]})[r]
        return (n ^ bfs[r]).count()
    f1, f7 = flips(1), flips(7)
    return f1 > f7, f"eps=1 flips {f1}; eps=7 flips {f7}"

R100 = None
@test("TC08", "Evaluation range", "EVAL")
def _():
    global R100
    R100 = run_cv(prepare(100, "corr_50", 7, seed=0), LR, seed=0)
    f = R100["LR"]["f1"]
    return 0.0 <= f <= 1.0, f"LR F1 = {f:.3f}  (100 rec, 50% corr, eps=7)"

@test("TC09", "Threshold baseline runs", "EVAL")
def _():
    f = R100["thr_fixed"]["f1"]
    return 0.0 <= f <= 1.0, f"Threshold F1 = {f:.3f}"

@test("TC10", "Model beats baseline (F-03)", "EVAL")
def _():
    r = run_cv(prepare(500, "corr_50", 5, seed=0), LR, seed=0)
    a, b = r["LR"]["f1"], r["thr_fixed"]["f1"]
    return a > b, f"LR {a:.3f} vs threshold {b:.3f}  (500 rec, eps=5)"

@test("TC11", "Pair feature size", "PPRL")
def _():
    it = iter(bfs.values()); x1, x2 = bf_to_np(next(it), L), bf_to_np(next(it), L)
    n = len(pair_features(x1, x2, "and_xor"))
    return n == 2 * L, f"feature length = {n}"

@test("TC12", "Text tokeniser cleaning (SR-04)", "TEXT")
def _():
    toks = tokenize("The patient had cancer and a biopsy. [[54.0, 'year']] M")
    return toks == ["cancer", "biopsy"], f"tokens = {toks}"

@test("TC13", "Bag-of-words counts", "TEXT")
def _():
    bow = bag_of_words("cancer cancer biopsy")
    return bow == {"cancer": 2, "biopsy": 1}, f"{bow}"

@test("TC14", "Counting filter self-similarity", "TEXT")
def _():
    v = CBF(L, K).encode("cancer cancer biopsy chemotherapy").reshape(1, -1)
    s = float(minsum_dice(v, v)[0])
    return abs(s - 1.0) < 1e-6, f"{s:.4f}"

@test("TC15", "Text DP noise non-negative (SR-04)", "TEXT")
def _():
    v = np.vstack([CBF(L, K).encode(n) for n in ["cancer biopsy", "fracture cast", "fever cough"]])
    m = float(add_dp_noise(v, 3, K).min())
    return m >= 0.0, f"min = {m}"

passed = sum(r[3] for r in results)
print("-" * 78 + f"\nResult: {passed} / {len(results)} test cases passed  ({time.time()-t0:.0f}s)")
sys.exit(0 if passed == len(results) else 1)
