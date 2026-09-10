"""
Print the LaTeX bodies of the benchmark-derived tables from the corrected data.

Tables 5, 7, 11, 12 and the "This work" rows of Table 21 all read from the
benchmark table, whose UNSW-NB15 multi-class block was recomputed after the
duplicated Backdoor labels were merged.  Rather than hand-edit several dozen
cells, this emits each body so it can be pasted in and then checked by
verify_manuscript_tables.py.

It only prints; it does not modify main.tex.

Usage:  python emit_corrected_tables.py [--table multi|degradation|ft|tuning|sota]
"""

import argparse

import pandas as pd

from utils import TABLES, REVISION

CORR = TABLES / "benchmark_results_corrected.csv"
TREES = ["XGBoost", "LightGBM", "RandomForest"]
DS3 = ["cicids2017", "ton_iot", "unsw_nb15"]
# printed name -> CSV name, in the manuscript's row order
ROWS = [("XGBoost", "XGBoost"), ("Random Forest", "RandomForest"),
        ("LightGBM", "LightGBM"), ("SVM", "SVM"), ("k-NN", "kNN"),
        ("MLP", "MLP"), ("1D-CNN", "CNN1D"), ("BiLSTM", "BiLSTM")]


def load():
    if not CORR.exists():
        raise SystemExit(f"{CORR.name} not found -- run "
                         "merge_corrected_benchmark.py first")
    return pd.read_csv(CORR)


def _f(x, dp=3):
    return f"{x:.{dp}f}"


def _thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def multi_results(b):
    """Table 5 body (macro F1 +- sd, accuracy, fit seconds)."""
    print("% ---- tab:multi_results ----")
    best = {ds: b[(b.task == "multi") & (b.dataset == ds)].f1_macro.max()
            for ds in DS3}
    for tex, m in ROWS:
        cells = []
        for ds in DS3:
            r = b[(b.model == m) & (b.dataset == ds) & (b.task == "multi")]
            if not len(r):
                cells += ["---", "---", "---"]
                continue
            r = r.iloc[0]
            f1 = f"{_f(r.f1_macro)}$\\pm${_f(r.f1_macro_std)}"
            if abs(r.f1_macro - best[ds]) < 1e-9:
                f1 = f"\\textbf{{{f1}}}"
            cells += [f1, _f(r.accuracy), _thousands(r.fit_seconds)]
        pad = " " * max(0, 13 - len(tex))
        print(f" {tex}{pad}& " + " & ".join(cells) + " \\\\")


def degradation(b):
    """Table 7 body: relative binary-to-multi-class drop, per dataset."""
    print("% ---- tab:degradation ----")
    for tex, m in ROWS:
        cells = []
        for ds in DS3:
            bn = b[(b.model == m) & (b.dataset == ds) & (b.task == "binary")]
            mu = b[(b.model == m) & (b.dataset == ds) & (b.task == "multi")]
            if not len(bn) or not len(mu):
                cells.append("---")
                continue
            d = (bn.f1_macro.iloc[0] - mu.f1_macro.iloc[0]) / bn.f1_macro.iloc[0]
            cells.append(f"{d * 100:.1f}")
        pad = " " * max(0, 13 - len(tex))
        print(f" {tex}{pad}& " + " & ".join(cells) + " \\\\")


def fttransformer(b):
    """Table 11 body: FT-Transformer against the sequence models and best tree."""
    ft = pd.read_csv(TABLES / "e7_ft_transformer.csv")
    print("% ---- tab:fttransformer ----")
    names = {"cicids2017": "CICIDS2017", "ton_iot": "ToN-IoT",
             "unsw_nb15": "UNSW-NB15"}
    for task, shown in (("binary", "binary"), ("multi", "multi-class")):
        for ds in DS3:
            g = ft[(ft.dataset == ds) & (ft.task == task)]
            cnn = b[(b.model == "CNN1D") & (b.dataset == ds) & (b.task == task)]
            lstm = b[(b.model == "BiLSTM") & (b.dataset == ds) & (b.task == task)]
            best = b[(b.model.isin(TREES)) & (b.dataset == ds)
                     & (b.task == task)].f1_macro.max()
            print(f"{names[ds]:10s} & {shown:11s} & {g.f1_macro.mean():.4f} & "
                  f"{cnn.f1_macro.iloc[0]:.4f} & {lstm.f1_macro.iloc[0]:.4f} & "
                  f"\\textbf{{{best:.4f}}} \\\\")
    # the prose figures that depend on this table
    print("\n% prose:")
    short, beats_cnn, beats_lstm = [], 0, 0
    for task in ("binary", "multi"):
        for ds in DS3:
            f = ft[(ft.dataset == ds) & (ft.task == task)].f1_macro.mean()
            best = b[(b.model.isin(TREES)) & (b.dataset == ds)
                     & (b.task == task)].f1_macro.max()
            cnn = b[(b.model == "CNN1D") & (b.dataset == ds)
                    & (b.task == task)].f1_macro.iloc[0]
            lstm = b[(b.model == "BiLSTM") & (b.dataset == ds)
                     & (b.task == task)].f1_macro.iloc[0]
            short.append(best - f)
            beats_cnn += f > cnn
            beats_lstm += f > lstm
    print(f"%   mean shortfall against best tree : {sum(short)/len(short):.4f}")
    print(f"%   beats 1D-CNN in {beats_cnn} of 6, BiLSTM in {beats_lstm} of 6")
    for ds in DS3:
        for task in ("binary", "multi"):
            f = ft[(ft.dataset == ds) & (ft.task == task)].f1_macro.mean()
            best = b[(b.model.isin(TREES)) & (b.dataset == ds)
                     & (b.task == task)].f1_macro.max()
            print(f"%   gap {ds}/{task}: {best - f:.4f}")


def tuning(b):
    """Table 12 body: fixed vs tuned sequence models."""
    tuned = pd.read_csv(TABLES / "e6_dl_tuned_folds.csv")
    print("% ---- tab:dltuning ----")
    names = {"ton_iot": "ToN-IoT", "unsw_nb15": "UNSW-NB15"}
    gains = []
    for ds in ("ton_iot", "unsw_nb15"):
        for tex, m in (("1D-CNN", "CNN1D"), ("BiLSTM", "BiLSTM")):
            fx = b[(b.model == m) & (b.dataset == ds)
                   & (b.task == "multi")].f1_macro.iloc[0]
            tu = tuned[(tuned.dataset == ds) & (tuned.model == m)
                       & (tuned.task == "multi")].f1_macro.mean()
            best = b[(b.model.isin(TREES)) & (b.dataset == ds)
                     & (b.task == "multi")].f1_macro.max()
            gains.append(tu - fx)
            print(f"{names[ds]:9s} & {tex:6s} & {fx:.4f} & {tu:.4f} & "
                  f"$+${tu - fx:.4f} & {best:.4f} & {best - tu:.4f} \\\\")
    print(f"\n% prose: mean gain across the four combinations = "
          f"{sum(gains)/len(gains):+.4f}")
    print(f"% largest single gain = {max(gains):+.4f}")


def sota(b):
    """Table 21 'This work' rows: best model and its binary/multi F1."""
    print("% ---- tab:sota (This work rows) ----")
    names = {"cicids2017": "CICIDS2017", "ton_iot": "ToN-IoT",
             "unsw_nb15": "UNSW-NB15"}
    for ds in DS3:
        bn = b[(b.dataset == ds) & (b.task == "binary")]
        mu = b[(b.dataset == ds) & (b.task == "multi")]
        bb = bn.loc[bn.f1_macro.idxmax()]
        bm = mu.loc[mu.f1_macro.idxmax()]
        note = (f"{bb.model}" if bb.model == bm.model
                else f"{bb.model} / {bm.model}")
        print(f"\\textbf{{This work}} & {names[ds]:10s} & {note:24s} & "
              f"{bb.f1_macro:.3f} & {bm.f1_macro:.3f} & \\textbf{{Yes}} & "
              f"5-fold CV \\\\")
        print(f"%   binary best = {bb.model} {bb.f1_macro:.4f}; "
              f"multi best = {bm.model} {bm.f1_macro:.4f}")


EMIT = {"multi": multi_results, "degradation": degradation,
        "ft": fttransformer, "tuning": tuning, "sota": sota}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(EMIT))
    a = ap.parse_args()
    bench = load()
    for key in ([a.table] if a.table else EMIT):
        EMIT[key](bench)
        print()
