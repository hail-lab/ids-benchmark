"""
Revision E2 — cross-dataset transfer on a manually aligned feature schema.

Addresses R6.3: the claim that "no universal feature set transfers across
datasets" must be tested directly.  The three datasets use different flow
exporters (CICFlowMeter / Zeek / Argus), so raw column names barely
intersect; here each dataset is mapped to a common semantic schema of five
base flow measurements plus derived statistics:

    duration_s, src_bytes, dst_bytes, src_pkts, dst_pkts,
    total_bytes, total_pkts, byte_rate, pkt_rate,
    mean_pkt_size, src_mean_pkt_size, dst_mean_pkt_size, fwd_byte_ratio

Protocol: train XGBoost and RandomForest on dataset A (binary task, full
common-schema features, min-max scaled on A), test on dataset B for all six
ordered pairs; the within-dataset 5-fold CV score on the same schema is the
reference ceiling.

Usage:  python e2_transfer.py
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from utils import TABLES, RANDOM_STATE, N_JOBS, dataset_path, log
from evaluation import compute_metrics

OUT = TABLES / "e2_cross_dataset_transfer.csv"
MAX_ROWS = 500_000  # per dataset; keeps 6 pairs tractable on CPU

# dataset → {common_name: source_column}
BASE_MAP = {
    "cicids2017": {
        "duration_s": "flow_duration",          # microseconds → seconds below
        "src_bytes": "total_length_of_fwd_packets",
        "dst_bytes": "total_length_of_bwd_packets",
        "src_pkts": "total_fwd_packets",
        "dst_pkts": "total_backward_packets",
    },
    "ton_iot": {
        "duration_s": "duration",
        "src_bytes": "src_bytes",
        "dst_bytes": "dst_bytes",
        "src_pkts": "src_pkts",
        "dst_pkts": "dst_pkts",
    },
    "unsw_nb15": {
        "duration_s": "dur",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "src_pkts": "spkts",
        "dst_pkts": "dpkts",
    },
}


def load_common(ds: str) -> pd.DataFrame:
    df = pd.read_parquet(
        dataset_path(ds),
        columns=list(BASE_MAP[ds].values()) + ["label_binary"],
    )
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce")
                        for k, v in BASE_MAP[ds].items()})
    if ds == "cicids2017":
        out["duration_s"] = out["duration_s"] / 1e6
    out = out.clip(lower=0)

    dur = out["duration_s"].replace(0, np.nan)
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["total_pkts"] = out["src_pkts"] + out["dst_pkts"]
    out["byte_rate"] = (out["total_bytes"] / dur).fillna(0)
    out["pkt_rate"] = (out["total_pkts"] / dur).fillna(0)
    tp = out["total_pkts"].replace(0, np.nan)
    out["mean_pkt_size"] = (out["total_bytes"] / tp).fillna(0)
    sp = out["src_pkts"].replace(0, np.nan)
    dp = out["dst_pkts"].replace(0, np.nan)
    out["src_mean_pkt_size"] = (out["src_bytes"] / sp).fillna(0)
    out["dst_mean_pkt_size"] = (out["dst_bytes"] / dp).fillna(0)
    tb = out["total_bytes"].replace(0, np.nan)
    out["fwd_byte_ratio"] = (out["src_bytes"] / tb).fillna(0)

    out = out.replace([np.inf, -np.inf], 0).astype(np.float32)
    out["label_binary"] = df["label_binary"].to_numpy()

    if len(out) > MAX_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        # stratified subsample
        idx = (out.groupby("label_binary", group_keys=False)
                  .apply(lambda g: g.sample(
                      max(1, int(len(g) * MAX_ROWS / len(out))),
                      random_state=rng))).index
        out = out.loc[idx].reset_index(drop=True)
    log.info("[%s] common schema: %d rows, %d features, attack rate %.3f",
             ds, len(out), out.shape[1] - 1, out["label_binary"].mean())
    return out


def _make_model(name):
    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                  n_jobs=N_JOBS)


def main():
    data = {ds: load_common(ds) for ds in BASE_MAP}
    feats = [c for c in data["ton_iot"].columns if c != "label_binary"]
    rows = []

    # Within-dataset ceiling on the common schema (5-fold CV)
    for ds, df in data.items():
        X = df[feats].to_numpy()
        y = df["label_binary"].to_numpy()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for mname in ("XGBoost", "RandomForest"):
            f1s = []
            for tr, te in skf.split(X, y):
                lo, hi = X[tr].min(0), X[tr].max(0)
                sc = np.where(hi - lo == 0, 1, hi - lo)
                model = _make_model(mname)
                model.fit((X[tr] - lo) / sc, y[tr])
                y_pred = model.predict((X[te] - lo) / sc)
                probs = model.predict_proba((X[te] - lo) / sc)
                f1s.append(compute_metrics(y[te], y_pred, probs, 2))
            avg = {k: round(float(np.mean([f[k] for f in f1s])), 4)
                   for k in f1s[0]}
            avg.update(train_dataset=ds, test_dataset=ds, model=mname,
                       setting="within (5-fold CV)", n_features=len(feats))
            rows.append(avg)
            log.info("WITHIN %s %s — F1=%.4f", ds, mname, avg["f1_macro"])

    # Cross-dataset: all ordered pairs
    for src, tgt in itertools.permutations(data.keys(), 2):
        Xs = data[src][feats].to_numpy()
        ys = data[src]["label_binary"].to_numpy()
        Xt = data[tgt][feats].to_numpy()
        yt = data[tgt]["label_binary"].to_numpy()
        lo, hi = Xs.min(0), Xs.max(0)
        sc = np.where(hi - lo == 0, 1, hi - lo)
        for mname in ("XGBoost", "RandomForest"):
            model = _make_model(mname)
            model.fit((Xs - lo) / sc, ys)
            y_pred = model.predict((Xt - lo) / sc)
            probs = model.predict_proba((Xt - lo) / sc)
            m = compute_metrics(yt, y_pred, probs, 2)
            m.update(train_dataset=src, test_dataset=tgt, model=mname,
                     setting="transfer", n_features=len(feats))
            rows.append(m)
            log.info("TRANSFER %s → %s %s — F1=%.4f", src, tgt, mname,
                     m["f1_macro"])

    pd.DataFrame(rows).to_csv(OUT, index=False)
    log.info("E2 done → %s", OUT)


if __name__ == "__main__":
    main()
