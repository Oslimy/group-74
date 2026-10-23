# quick_demo.py - fast single-setting demo. Prints a results table for all 5 models.
# Run:  python3 quick_demo.py
import warnings, time; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from ml_linkage import prepare, run_cv

SIZE, CORR, EPS = 100, "corr_50", 7     # one fast setting
models = {
 "LR":  lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
 "RF":  lambda: RandomForestClassifier(n_estimators=60, class_weight="balanced", n_jobs=-1, random_state=0),
 "SVM": lambda: LinearSVC(class_weight="balanced", max_iter=5000),
 "XGB": lambda: XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.3, n_jobs=-1, eval_metric="logloss", verbosity=0),
}
print(f"Running {SIZE} records, {CORR}, epsilon={EPS} ... (about a minute)\n", flush=True)
t = time.time()
P = prepare(SIZE, CORR, EPS, seed=0)
r = run_cv(P, models, seed=0)

print(f"{'Method':<24}{'F1':>8}{'Precision':>12}{'Recall':>9}")
print("-" * 53)
labels = {"thr_fixed":"Threshold 0.8 (baseline)","thr_tuned":"Threshold (tuned)","LR":"Logistic Regression","RF":"Random Forest","SVM":"SVM (linear)","XGB":"XGBoost"}
for m in ["thr_fixed","thr_tuned","LR","RF","SVM","XGB"]:
    print(f"{labels[m]:<24}{r[m]['f1']:>8.3f}{r[m]['prec']:>12.3f}{r[m]['rec']:>9.3f}", flush=True)
print(f"\nFinished in {time.time()-t:.0f}s. Higher F1 = better. Note how the fixed 0.8 baseline lags the learned models.")
