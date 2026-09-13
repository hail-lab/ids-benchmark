"""
Verify that every number in the revision's tables matches its source CSV.

Each check recomputes a table's cells from the released results and compares
against the values parsed out of main.tex.  This exists because two of the
errors found during the pre-submission audit were exactly this kind: a cell
that no longer matched the CSV behind it (a LinearSVC mean taken over two of
three folds) and a whole table that had never been regenerated after its
experiment was re-run (the nested feature-selection ablation).

Run it after any change to the result tables or the manuscript:

    python verify_manuscript_tables.py
"""

import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from utils import TABLES, REVISION, log

MAIN = REVISION / "manuscript" / "main.tex"
TOL = 0.00006          # values are printed to 4 dp

failures: list[str] = []
checks = 0


def body(label: str) -> str:
    """The LaTeX between a table's label and the end of its float."""
    tex = MAIN.read_text(encoding="utf-8")
    m = re.search(r"\\label\{" + label + r"\}(.*?)\\end\{table\*?\}", tex, re.S)
    if not m:
        failures.append(f"{label}: not found in main.tex")
        return ""
    return m.group(1)


def cells(line: str) -> list[str]:
    out = []
    for c in line.split("&"):
        c = re.sub(r"\\(textbf|rev|emph|textit|mathbf)\{", "", c)
        c = c.replace("{,}", "").replace("\\\\", "").replace("}", "")
        c = c.replace("$\\dagger$", "").replace("\\hline", "")
        out.append(c.strip())
    return out


def num(s: str):
    # thousands separators appear both as 1{,}061 (stripped in cells()) and as
    # plain 1,061; without removing the comma the regex reads only the "1"
    s = s.replace("$", "").replace("+", "").replace(",", "")
    m = re.search(r"-?\d+\.\d+|-?\d+", s)
    return float(m.group(0)) if m else None


def round_half_up(x: float, dp: int) -> float:
    """Round the way the manuscript's printed values were rounded.

    Python's ``round`` is banker's rounding, so round(0.8935, 3) gives 0.893
    while the paper prints 0.894.  Comparing against it produces spurious
    0.001 mismatches on exactly the cells that sit on a tie.
    """
    q = Decimal(1).scaleb(-dp)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def eq(label: str, what: str, paper, csv, tol=TOL):
    global checks
    checks += 1
    if paper is None or csv is None:
        failures.append(f"{label}: {what}: could not parse (paper={paper}, csv={csv})")
        return
    if abs(paper - csv) > tol:
        failures.append(f"{label}: {what}: paper={paper} csv={csv:.4f} "
                        f"(diff {paper - csv:+.4f})")


# ---------------------------------------------------------------- Table 6
def check_splits():
    d = pd.read_csv(TABLES / "e1_split_schemes.csv")
    d["key"] = d.dataset + "/" + d.task
    scheme_of = {"Stratified": "stratified", "Day-grouped": "group",
                 "Host-grouped": "group", "Temporal": "temporal"}
    order = ["cicids2017/binary", "cicids2017/multi", "ton_iot/binary",
             "ton_iot/multi", "unsw_nb15/binary", "unsw_nb15/multi"]
    idx = -1
    seen: set[str] = set()
    for line in body("tab:splits").split("\n"):
        c = cells(line)
        if len(c) < 5 or c[1] not in scheme_of:
            continue
        if c[1] == "Stratified":
            idx += 1
        key = order[idx]
        g = d[(d.key == key) & (d.scheme == scheme_of[c[1]])]
        eq("tab:splits", f"{key} {c[1]} F1", num(c[2]), g.f1_macro.mean())
        eq("tab:splits", f"{key} {c[1]} SD", num(c[3]), g.f1_macro.std())
        eq("tab:splits", f"{key} {c[1]} excl", num(c[4]),
           g.excluded_test_frac.mean() * 100, tol=0.06)
        seen.add(f"{key}/{c[1]}")
    if len(seen) != 16:
        failures.append(f"tab:splits: parsed {len(seen)} rows, expected 16")


# ---------------------------------------------------------------- Table 8
def check_lgbmsens():
    d = pd.read_csv(TABLES / "e4_lgbm_sensitivity.csv")
    # matched on distinctive substrings: the LaTeX row labels carry \texttt{}
    # and escaped underscores, so prefix matching on the printed name is brittle
    name = [("balanced weights", "ova+weight"),
            ("One-vs-all objective", "ova"),
            ("class", "class_weight"),
            ("SMOTE", "smote"),
            ("Default", "default"),
            ("Tuned", "tuned"),
            ("undersampling", "undersample")]
    hits = 0
    for line in body("tab:lgbmsens").split("\n"):
        c = cells(line)
        if len(c) < 4:
            continue
        cfg = next((v for k, v in name if k in c[0]), None)
        if not cfg:
            continue
        g = d[d.config == cfg]
        eq("tab:lgbmsens", f"{cfg} F1", num(c[1]), g.f1_macro.mean())
        eq("tab:lgbmsens", f"{cfg} SD", num(c[2]), g.f1_macro.std(), tol=0.0006)
        eq("tab:lgbmsens", f"{cfg} bal", num(c[3]),
           g.balanced_accuracy.mean(), tol=0.0006)
        hits += 1
    if hits != 7:
        failures.append(f"tab:lgbmsens: parsed {hits} rows, expected 7")


# ---------------------------------------------------------------- Table 10
def check_imbalance():
    d = pd.read_csv(TABLES / "e5_class_weight.csv")
    ds_of = {"CICIDS2017": "cicids2017", "ToN-IoT": "ton_iot",
             "UNSW-NB15": "unsw_nb15"}
    hits = 0
    for line in body("tab:imbalance").split("\n"):
        c = cells(line)
        if len(c) < 5 or c[0] not in ds_of:
            continue
        ds = ds_of[c[0]]
        task = "binary" if c[1].startswith("binary") else "multi"
        for col, model in ((2, "RandomForest"), (3, "SVM"), (4, "MLP")):
            s = d[(d.dataset == ds) & (d.task == task) & (d.model == model)]
            p = s.pivot_table(index="fold", columns="config", values="f1_macro")
            arm = "weighted_loss" if model == "MLP" else "balanced_resample"
            delta = (p[arm] - p["default"]).mean()
            paper = num(c[col].replace("$-$", "-"))
            eq("tab:imbalance", f"{ds}/{task}/{model}", paper, delta)
        hits += 1
    if hits != 6:
        failures.append(f"tab:imbalance: parsed {hits} rows, expected 6")


# ---------------------------------------------------------------- Table 14
def check_latency():
    d = pd.read_csv(TABLES / "e8a_inference_latency.csv")
    b = d[d.task == "binary"].groupby("model")
    name = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP": "MLP",
            "1D-CNN": "CNN1D", "Random Forest": "RandomForest",
            "BiLSTM": "BiLSTM", "SVM": "SVM", "k-NN": "kNN"}
    hits = 0
    for line in body("tab:latency").split("\n"):
        c = cells(line)
        if len(c) < 3 or c[0] not in name:
            continue
        m = name[c[0]]
        eq("tab:latency", f"{m} batch", num(c[1]),
           b.batch_us_per_sample.mean()[m], tol=0.06)
        eq("tab:latency", f"{m} single", num(c[2]),
           b.single_ms_median.mean()[m], tol=0.006)
        hits += 1
    if hits != 8:
        failures.append(f"tab:latency: parsed {hits} rows, expected 8")


# ---------------------------------------------------------------- Table 15
def check_ablation():
    d = pd.read_csv(TABLES / "e3_nested_fs.csv")
    st = pd.read_csv(TABLES / "e3_selection_stability.csv")
    cfg_of = {"No FS": "No FS", "MI only (top 30)": "MI only (top 30)",
              "RF only (top 15)": "RF only (top 15)",
              "Hybrid MI$\\to$RF": "Hybrid MI-RF (15)"}
    ds = ["cicids2017", "ton_iot", "unsw_nb15"]
    seen_f1, seen_jac = 0, 0
    for line in body("tab:ablation").split("\n"):
        c = cells(line)
        if len(c) < 5:
            continue
        cfg = cfg_of.get(c[0])
        if not cfg:
            continue
        # the F1 block prints the native feature counts; the Jaccard block
        # repeats each selector with its own count, so distinguish by value
        vals = [num(c[2 + i]) for i in range(3)]
        if all(v is not None and v > 0.9 for v in vals) and seen_f1 < 4:
            for i, dsn in enumerate(ds):
                g = d[(d.dataset == dsn) & (d.config == cfg)]
                eq("tab:ablation", f"{dsn}/{cfg} F1", vals[i], g.f1_macro.mean())
            seen_f1 += 1
        else:
            for i, dsn in enumerate(ds):
                g = st[(st.dataset == dsn) & (st.config == cfg)]
                if len(g):
                    eq("tab:ablation", f"{dsn}/{cfg} jaccard", vals[i],
                       float(g.mean_jaccard.iloc[0]), tol=0.0006)
            seen_jac += 1
    if seen_f1 != 4:
        failures.append(f"tab:ablation: parsed {seen_f1} F1 rows, expected 4")
    if seen_jac != 3:
        failures.append(f"tab:ablation: parsed {seen_jac} Jaccard rows, expected 3")


# ---------------------------------------------------------------- Table 16
def check_transfer():
    d = pd.read_csv(TABLES / "e2_cross_dataset_transfer.csv")
    ds_of = {"CICIDS2017": "cicids2017", "ToN-IoT": "ton_iot",
             "UNSW-NB15": "unsw_nb15"}
    cols = ["cicids2017", "ton_iot", "unsw_nb15"]
    models = ["XGBoost", "RandomForest"]
    block, hits = -1, 0
    for line in body("tab:transfer").split("\n"):
        c = cells(line)
        if len(c) < 5 or c[1] not in ds_of:
            continue
        if c[1] == "CICIDS2017":
            block += 1
        model = models[min(block, 1)]
        train = ds_of[c[1]]
        for i, test in enumerate(cols):
            cell = c[2 + i]
            f1 = num(cell)
            bal = num(cell.split("(")[1]) if "(" in cell else None
            g = d[(d.train_dataset == train) & (d.test_dataset == test)
                  & (d.model == model)]
            if not len(g):
                failures.append(f"tab:transfer: no CSV row {train}->{test}/{model}")
                continue
            eq("tab:transfer", f"{model} {train}->{test} F1", f1,
               float(g.f1_macro.iloc[0]), tol=0.0006)
            eq("tab:transfer", f"{model} {train}->{test} bal", bal,
               float(g.balanced_accuracy.iloc[0]), tol=0.0006)
        hits += 1
    if hits != 6:
        failures.append(f"tab:transfer: parsed {hits} rows, expected 6")


# ---------------------------------------------------------------- Table 20
def check_foldcounts():
    d = pd.read_csv(TABLES / "e8c_per_class_fold_counts.csv")
    hits = 0
    for line in body("tab:foldcounts").split("\n"):
        c = cells(line)
        if len(c) < 5:
            continue
        total, mn, mx = num(c[2]), num(c[3]), num(c[4])
        if total is None or mn is None:
            continue
        g = d[d.n_total == int(total)]
        if not len(g):
            failures.append(f"tab:foldcounts: no class with {int(total)} samples")
            continue
        eq("tab:foldcounts", f"{c[1]} min", mn, float(g.min_per_fold.iloc[0]), tol=0.5)
        eq("tab:foldcounts", f"{c[1]} max", mx, float(g.max_per_fold.iloc[0]), tol=0.5)
        hits += 1
    if hits != 14:
        failures.append(f"tab:foldcounts: parsed {hits} rows, expected 14")


# ------------------------------------------------ the benchmark-derived tables
# Tables 4, 5, 7, 11 and 12 all read from the main benchmark table.  Prefer the
# label-corrected version when it exists, since the UNSW-NB15 multi-class block
# in the original describes an 11-class task.
def _benchmark() -> pd.DataFrame:
    corrected = TABLES / "benchmark_results_corrected.csv"
    src = corrected if corrected.exists() else (
        REVISION.parent / "outputs" / "tables" / "benchmark_results.csv")
    log.info("benchmark table: %s", src.name)
    return pd.read_csv(src)


MODEL_TEX = {"XGBoost": "XGBoost", "Random Forest": "RandomForest",
             "LightGBM": "LightGBM", "SVM": "SVM", "k-NN": "kNN", "MLP": "MLP",
             "1D-CNN": "CNN1D", "BiLSTM": "BiLSTM"}
DS3 = ["cicids2017", "ton_iot", "unsw_nb15"]


def _check_results(label: str, task: str):
    b = _benchmark()
    hits = 0
    for line in body(label).split("\n"):
        c = cells(line)
        if len(c) < 10 or c[0] not in MODEL_TEX:
            continue
        m = MODEL_TEX[c[0]]
        for i, ds in enumerate(DS3):
            paper = num(c[1 + i * 3].split("\\pm")[0])
            row = b[(b.model == m) & (b.dataset == ds) & (b.task == task)]
            if not len(row):
                failures.append(f"{label}: no CSV row {m}/{ds}/{task}")
                continue
            csv = float(row.f1_macro.iloc[0])
            dp = len(str(paper).split(".")[1]) if "." in str(paper) else 4
            eq(label, f"{m}/{ds}", paper, round_half_up(csv, dp))
        hits += 1
    if hits != 8:
        failures.append(f"{label}: parsed {hits} rows, expected 8")


def check_binary_results():
    _check_results("tab:binary_results", "binary")


def check_multi_results():
    _check_results("tab:multi_results", "multi")


def check_degradation():
    b = _benchmark()
    hits = 0
    for line in body("tab:degradation").split("\n"):
        c = cells(line)
        if len(c) < 4 or c[0] not in MODEL_TEX:
            continue
        m = MODEL_TEX[c[0]]
        for i, ds in enumerate(DS3):
            paper = num(c[1 + i])
            bn = b[(b.model == m) & (b.dataset == ds) & (b.task == "binary")]
            mu = b[(b.model == m) & (b.dataset == ds) & (b.task == "multi")]
            if not len(bn) or not len(mu):
                failures.append(f"tab:degradation: missing {m}/{ds}")
                continue
            calc = ((bn.f1_macro.iloc[0] - mu.f1_macro.iloc[0])
                    / bn.f1_macro.iloc[0] * 100)
            eq("tab:degradation", f"{m}/{ds}", paper, calc, tol=0.06)
        hits += 1
    if hits != 8:
        failures.append(f"tab:degradation: parsed {hits} rows, expected 8")


def check_fttransformer():
    ft = pd.read_csv(TABLES / "e7_ft_transformer.csv")
    b = _benchmark()
    trees = ["XGBoost", "LightGBM", "RandomForest"]
    ds_of = {"CICIDS2017": "cicids2017", "ToN-IoT": "ton_iot",
             "UNSW-NB15": "unsw_nb15"}
    hits = 0
    for line in body("tab:fttransformer").split("\n"):
        c = cells(line)
        if len(c) < 6 or c[0] not in ds_of:
            continue
        ds = ds_of[c[0]]
        task = "binary" if c[1].startswith("binary") else "multi"
        g = ft[(ft.dataset == ds) & (ft.task == task)]
        eq("tab:fttransformer", f"{ds}/{task} FT", num(c[2]), g.f1_macro.mean())
        for col, m in ((3, "CNN1D"), (4, "BiLSTM")):
            row = b[(b.model == m) & (b.dataset == ds) & (b.task == task)]
            eq("tab:fttransformer", f"{ds}/{task} {m}", num(c[col]),
               float(row.f1_macro.iloc[0]), tol=0.0006)
        best = b[(b.model.isin(trees)) & (b.dataset == ds)
                 & (b.task == task)].f1_macro.max()
        eq("tab:fttransformer", f"{ds}/{task} best tree", num(c[5]), best,
           tol=0.0006)
        hits += 1
    if hits != 6:
        failures.append(f"tab:fttransformer: parsed {hits} rows, expected 6")


def check_dltuning():
    tuned = pd.read_csv(TABLES / "e6_dl_tuned_folds.csv")
    b = _benchmark()
    trees = ["XGBoost", "LightGBM", "RandomForest"]
    ds_of = {"ToN-IoT": "ton_iot", "UNSW-NB15": "unsw_nb15"}
    m_of = {"1D-CNN": "CNN1D", "BiLSTM": "BiLSTM"}
    hits = 0
    for line in body("tab:dltuning").split("\n"):
        c = cells(line)
        if len(c) < 7 or c[0] not in ds_of or c[1] not in m_of:
            continue
        ds, m = ds_of[c[0]], m_of[c[1]]
        fixed = b[(b.model == m) & (b.dataset == ds) & (b.task == "multi")]
        eq("tab:dltuning", f"{ds}/{m} fixed", num(c[2]),
           float(fixed.f1_macro.iloc[0]), tol=0.0006)
        g = tuned[(tuned.dataset == ds) & (tuned.model == m)
                  & (tuned.task == "multi")]
        eq("tab:dltuning", f"{ds}/{m} tuned", num(c[3]), g.f1_macro.mean())
        eq("tab:dltuning", f"{ds}/{m} gain", num(c[4].replace("$-$", "-")),
           g.f1_macro.mean() - float(fixed.f1_macro.iloc[0]))
        best = b[(b.model.isin(trees)) & (b.dataset == ds)
                 & (b.task == "multi")].f1_macro.max()
        eq("tab:dltuning", f"{ds}/{m} best tree", num(c[5]), best, tol=0.0006)
        eq("tab:dltuning", f"{ds}/{m} gap", num(c[6]),
           best - g.f1_macro.mean(), tol=0.0006)
        hits += 1
    if hits != 4:
        failures.append(f"tab:dltuning: parsed {hits} rows, expected 4")


# ---------------------------------------------------------------- Table 9
def check_subsample():
    """Subsample / iteration-budget study, scikit-learn.

    Rows are keyed by (model, config, intended cap). The cap matters for k-NN,
    whose three sizes are three experiments, but must not be taken from the
    raw n_train for LinearSVC, whose folds differ by the remainder of an
    integer split -- grouping on that produced a mean over two of three folds
    in the superseded table.
    """
    src = TABLES / "e5_subsample_curve_sklearn.csv"
    if not src.exists():
        failures.append("tab:subsample: scikit-learn results missing")
        return
    d = pd.read_csv(src)
    caps = (50_000, 100_000, 200_000)
    d["cap"] = d.n_train.map(lambda n: next((c for c in caps if n == c), "full"))
    g = d.groupby(["dataset", "model", "config", "cap"]).agg(
        f1=("f1_macro", "mean"), conv=("converged", "all"))

    ds_order = ["cicids2017", "unsw_nb15"]
    idx = -1
    hits = 0
    for line in body("tab:subsample").split("\n"):
        c = cells(line)
        if len(c) < 5:
            continue
        name, rows_txt = c[1], c[2]
        if "k-NN" in name:
            model, config = "kNN", "k5"
        elif "LinearSVC" in name:
            model, config = "LinearSVC", "linear"
        elif "max_iter" in name or "max\\_iter" in name:
            model = "SVM"
            tail = name.split("=")[-1].strip()
            budget = "50000" if tail.startswith("50000") else "5000"
            config = f"rbf_maxiter{budget}"
        else:
            continue
        if model == "kNN" and num(rows_txt) == 50_000:
            idx += 1                       # first row of a dataset block
        ds = ds_order[max(idx, 0)]

        n = num(rows_txt)
        cap = "full" if model == "LinearSVC" else int(n)
        if model == "SVM" and cap == 100_000:
            config = "rbf_maxiter5000_100k"
        key = (ds, model, config, cap)
        if key not in g.index:
            failures.append(f"tab:subsample: no CSV row for {key}")
            continue
        eq("tab:subsample", f"{ds}/{model}/{config}/{cap}", num(c[3]),
           float(g.loc[key, "f1"]))
        hits += 1
    if hits != 14:
        failures.append(f"tab:subsample: parsed {hits} rows, expected 14")


CHECKS = [check_splits, check_lgbmsens, check_imbalance, check_latency,
          check_ablation, check_transfer, check_foldcounts,
          check_binary_results, check_multi_results, check_degradation,
          check_fttransformer, check_dltuning, check_subsample]


if __name__ == "__main__":
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")

    print(f"\n{checks} cell comparison(s), {len(failures)} problem(s)")
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        sys.exit(1)
    log.info("all manuscript tables agree with their source CSVs")
