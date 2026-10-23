import warnings, time, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from text_encode import CBF, minsum_dice, add_dp_noise
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

U = "/mnt/user-data/uploads/"
rng = np.random.default_rng(0)

def load(split, n):
    A = pd.read_csv(U + ("1789810418627_train_hospital_A.csv" if split=="train" else "1789810418628_test_hospital_A.csv"))
    B = pd.read_csv(U + ("1789810418628_train_hospital_B.csv" if split=="train" else "1789810418628_test_hospital_B.csv"))
    L = pd.read_csv(U + ("1789810418628_train_labels.csv" if split=="train" else "1789810418629_test_labels.csv"))
    idx = rng.choice(len(A), size=min(n, len(A)), replace=False)
    return A.iloc[idx].reset_index(drop=True), B.iloc[idx].reset_index(drop=True), L.iloc[idx].reset_index(drop=True)

N_TRAIN, N_TEST = 8000, 4000
L, K = 1000, 10
t0 = time.time()
trA, trB, trL = load("train", N_TRAIN)
teA, teB, teL = load("test", N_TEST)
ytr, yte = trL.label.values, teL.label.values
print(f"loaded train {len(trA)} test {len(teA)} | pos rate tr {ytr.mean():.2f} te {yte.mean():.2f}")

enc = CBF(L, K)
def encode_col(series):
    return np.vstack([enc.encode(x) for x in series])
CtrA, CtrB = encode_col(trA.note), encode_col(trB.note)
CteA, CteB = encode_col(teA.note), encode_col(teB.note)
print(f"encoded CBFs ({time.time()-t0:.0f}s), vocab cached {len(enc._cache)}")

def pair_feats(A, B):
    return np.hstack([np.minimum(A, B), np.abs(A - B)])

rows = []
for eps in (None, 7, 3):
    # apply DP noise to all filters at this epsilon
    nA, nB = add_dp_noise(CtrA, eps, K), add_dp_noise(CtrB, eps, K)
    nteA, nteB = add_dp_noise(CteA, eps, K), add_dp_noise(CteB, eps, K)
    # --- threshold on min-sum Dice (tuned on train) ---
    str_ = minsum_dice(nA, nB); ste = minsum_dice(nteA, nteB)
    best_t, best_f = 0.0, -1
    for t in np.linspace(0.05, 0.8, 76):
        f = f1_score(ytr, (str_ >= t).astype(int))
        if f > best_f: best_f, best_t = f, t
    p = (ste >= best_t).astype(int)
    rows.append(dict(eps="inf" if eps is None else eps, model="Threshold (tuned)",
                     prec=precision_score(yte,p), rec=recall_score(yte,p), f1=f1_score(yte,p)))
    # --- ML models on pair features ---
    Xtr, Xte = pair_feats(nA, nB), pair_feats(nteA, nteB)
    for name, mk in [("Logistic Regression", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                     ("SVM (linear)", lambda: LinearSVC(class_weight="balanced", max_iter=5000))]:
        m = mk(); m.fit(Xtr, ytr); pr = m.predict(Xte)
        rows.append(dict(eps="inf" if eps is None else eps, model=name,
                         prec=precision_score(yte,pr), rec=recall_score(yte,pr), f1=f1_score(yte,pr)))
    print(f"eps={eps} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv("text_results.csv", index=False)
print("\n=== TEXT DATA RESULTS (counting Bloom filter, min-sum Dice) ===")
print(res.pivot_table(index="eps", columns="model", values="f1", aggfunc="first").round(3).to_string())
print(f"\nTOTAL {time.time()-t0:.0f}s")
