"""
exp5_complete.py
================
EXP5 — Hardware Drift Validation (self-contained, Qiskit + Matplotlib)

All logic in one file:
  - Qiskit callables (make_coupling_map, surrogates, compile_and_extract_features)
  - Dataset builders (boundary, scale)
  - Drift augmentation
  - Experiment sweep
  - Results PNG output

Usage
-----
    python exp5_complete.py

Output
------
    exp5_results.png   — multi-panel results figure
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, TwoLocal, EfficientSU2
from qiskit.transpiler import CouplingMap


# ════════════════════════════════════════════════════════════════════════════
# 1.  SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundaryDataset:
    df: pd.DataFrame

@dataclass
class ScaleDataset:
    df: pd.DataFrame


# ════════════════════════════════════════════════════════════════════════════
# 2.  QISKIT CALLABLES
# ════════════════════════════════════════════════════════════════════════════

def make_coupling_map(num_qubits: int, topology_family: str) -> CouplingMap:
    if topology_family == "line":
        return CouplingMap.from_line(num_qubits, bidirectional=True)
    elif topology_family == "ladder":
        cols  = max(2, (num_qubits + 1) // 2)
        cmap  = CouplingMap.from_grid(2, cols, bidirectional=True)
        edges = [(u, v) for u, v in cmap.get_edges() if u < num_qubits and v < num_qubits]
        return CouplingMap(couplinglist=edges)
    elif topology_family == "grid":
        side = max(2, int(np.ceil(num_qubits ** 0.5)))
        return CouplingMap.from_grid(side, side, bidirectional=True)
    else:
        raise ValueError(f"Unknown topology_family: {topology_family!r}")


def generate_pde_surrogate(
    num_qubits: int,
    boundary_condition: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """1-D PDE stencil surrogate. Dirichlet = open chain, Periodic = wrap-around RZZ."""
    rng = np.random.default_rng(seed)
    qc  = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for _ in range(n_steps):
        for q in range(num_qubits):
            qc.rx(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(num_qubits - 1):
            qc.rzz(rng.uniform(0, np.pi), q, q + 1)
        if boundary_condition == "periodic" and num_qubits > 2:
            qc.rzz(rng.uniform(0, np.pi), num_qubits - 1, 0)
    qc.measure_all()
    return qc


def generate_scale_surrogate(
    num_qubits: int,
    template_family: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """Scale-sweep surrogate. template_family ∈ {qft, twolocal, efficient, random}."""
    rng = np.random.default_rng(seed)

    if template_family == "qft":
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qft_c = QFT(num_qubits, do_swaps=True, approximation_degree=0)
        for _ in range(n_steps):
            qc.compose(qft_c, inplace=True)
        qc.measure_all()
        return qc

    elif template_family == "twolocal":
        n_params = num_qubits * n_steps * 2
        tl = TwoLocal(num_qubits, ["ry", "rz"], "cx", entanglement="linear", reps=n_steps)
        qc = tl.assign_parameters(rng.uniform(0, 2 * np.pi, tl.num_parameters))
        qc.measure_all()
        return qc

    elif template_family == "efficient":
        eff = EfficientSU2(num_qubits, reps=n_steps)
        qc  = eff.assign_parameters(rng.uniform(0, 2 * np.pi, eff.num_parameters))
        out = QuantumCircuit(num_qubits)
        out.compose(qc, inplace=True)
        out.measure_all()
        return out

    else:  # "random"
        qc = QuantumCircuit(num_qubits)
        for _ in range(n_steps):
            for q in range(num_qubits):
                qc.rx(rng.uniform(0, 2 * np.pi), q)
                qc.ry(rng.uniform(0, 2 * np.pi), q)
            for q in range(0, num_qubits - 1, 2):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc


def compile_and_extract_features(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    seed: int,
) -> dict:
    """Transpile onto coupling_map and return feature dict."""
    qc_nm = qc.remove_final_measurements(inplace=False)
    logical_depth     = qc_nm.depth()
    logical_twoq      = sum(1 for i in qc_nm.data if i.operation.num_qubits == 2)
    logical_total_ops = sum(1 for i in qc_nm.data if i.operation.num_qubits >= 1)

    t0 = time.perf_counter()
    transpiled = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=["cx", "u", "measure"],
        optimization_level=1,
        seed_transpiler=seed,
    )
    transpile_ms = (time.perf_counter() - t0) * 1000.0

    ops          = transpiled.count_ops()
    routed_cx    = ops.get("cx", 0)
    routed_swap  = ops.get("swap", 0)
    routed_depth = transpiled.depth()

    extra_cx    = max(0, routed_cx - logical_twoq)
    swap_equiv  = extra_cx / 3.0
    extra_twoq  = max(0, routed_cx + routed_swap - logical_twoq)
    extra_depth = max(0, routed_depth - logical_depth)

    routed_twoq      = routed_cx + routed_swap
    routed_total_ops = sum(v for k, v in ops.items() if k not in ("measure", "barrier", "reset"))

    sl = max(logical_twoq, 1)
    sd = max(logical_depth, 1)
    st = max(routed_twoq, 1)
    so = max(routed_total_ops, 1)

    return {
        "swap_equiv":        swap_equiv,
        "extra_twoq":        float(extra_twoq),
        "routed_depth":      float(routed_depth),
        "extra_depth":       float(extra_depth),
        "transpile_ms":      transpile_ms,
        "logical_depth":     float(logical_depth),
        "logical_twoq":      float(logical_twoq),
        "logical_total_ops": float(logical_total_ops),
        "depth_overhead":    routed_depth / sd,
        "twoq_overhead":     routed_twoq  / sl,
        "swap_fraction":     float(np.clip(swap_equiv / st,      0.0, 1.0)),
        "cx_fraction":       float(np.clip(routed_twoq / so,     0.0, 1.0)),
        "routed_cx":         float(routed_cx),
        "routed_total_ops":  float(routed_total_ops),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  DATASET BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def build_boundary_dataset(
    topology_family: str,
    num_qubits: int,
    n_samples_per_class: int,
    n_steps: int,
    seed: int,
) -> BoundaryDataset:
    rng  = np.random.default_rng(seed)
    cmap = make_coupling_map(num_qubits, topology_family)
    rows = []

    for idx in range(n_samples_per_class):
        lseed  = int(rng.integers(0, 10_000_000))
        ts_dir = int(rng.integers(0, 10_000_000))
        ts_per = int(rng.integers(0, 10_000_000))
        pair   = f"{topology_family}_bnd_{idx:05d}"

        for bc, lbl, ts in [("dirichlet", 0, ts_dir), ("periodic", 1, ts_per)]:
            qc   = generate_pde_surrogate(num_qubits, bc, n_steps, lseed)
            feat = compile_and_extract_features(qc, cmap, ts)
            feat.update({
                "task": "boundary", "label": lbl, "label_name": bc,
                "boundary": bc, "topology_family": topology_family,
                "pair_id": pair, "local_sample_idx": idx,
                "logical_seed": lseed, "transpile_seed": ts,
                "num_qubits": num_qubits, "n_steps": n_steps,
            })
            rows.append(feat)

    return BoundaryDataset(df=pd.DataFrame(rows))


def build_scale_dataset(
    resolutions: List[int],
    n_samples_per_res: int,
    topology_families: List[str],
    template_families: List[str],
    n_steps: int,
    seed: int,
) -> ScaleDataset:
    rng  = np.random.default_rng(seed)
    rows = []

    for topo in topology_families:
        for tmpl in template_families:
            for N in resolutions:
                cmap = make_coupling_map(N, topo)
                for si in range(n_samples_per_res):
                    lseed = int(rng.integers(0, 10_000_000))
                    tseed = int(rng.integers(0, 10_000_000))
                    qc    = generate_scale_surrogate(N, tmpl, n_steps, lseed)
                    feat  = compile_and_extract_features(qc, cmap, tseed)
                    feat.update({
                        "task": "scale", "label": N, "label_name": str(N), "N": N,
                        "instance_id": f"scale_{topo}_{tmpl}_N{N}_{si:05d}",
                        "local_sample_idx": si, "template_family": tmpl,
                        "topology_family": topo, "logical_seed": lseed,
                        "transpile_seed": tseed, "n_steps": n_steps,
                    })
                    rows.append(feat)

    return ScaleDataset(df=pd.DataFrame(rows))


# ════════════════════════════════════════════════════════════════════════════
# 4.  DRIFT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def augment_operational_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    sched = (6.0 * out["routed_depth"].values + 8.0 * out["extra_twoq"].values
             + rng.normal(0.0, 10.0, len(out)))
    idle  = (0.015 * out["routed_depth"].values + 0.10 * out["swap_equiv"].values
             + 0.06 * out["extra_depth"].values + rng.normal(0.0, 0.6, len(out)))
    out["sched_duration_ms"] = np.maximum(1.0, sched)
    out["idle_variance"]     = np.maximum(0.0, idle)
    return out


def apply_hardware_drift(
    df: pd.DataFrame,
    severity: float,
    topology_family: str,
    seed: int,
) -> pd.DataFrame:
    if severity == 0.0:
        return df.copy()

    rng       = np.random.default_rng(seed)
    out       = df.copy()
    topo_gain = 1.10 if topology_family == "line" else 1.00

    rl  = out["extra_twoq"].values   / max(float(out["extra_twoq"].max()),   1.0)
    dl  = out["routed_depth"].values / max(float(out["routed_depth"].max()), 1.0)
    tl  = out["transpile_ms"].values / max(float(out["transpile_ms"].max()), 1.0)

    out["swap_equiv"]   = np.maximum(0.0, out["swap_equiv"].values
        * (1 + topo_gain * severity * 0.18 * rl + rng.normal(0, 0.05 * severity, len(out))))
    out["extra_twoq"]   = np.maximum(0.0, out["extra_twoq"].values
        * (1 + topo_gain * severity * 0.20 * rl + rng.normal(0, 0.06 * severity, len(out))))
    out["routed_depth"] = np.maximum(1.0, out["routed_depth"].values
        * (1 + topo_gain * severity * (0.10 * rl + 0.08 * dl) + rng.normal(0, 0.04 * severity, len(out))))
    out["extra_depth"]  = np.maximum(0.0, out["extra_depth"].values
        * (1 + topo_gain * severity * 0.15 * dl + rng.normal(0, 0.06 * severity, len(out))))
    out["transpile_ms"] = np.maximum(1.0, out["transpile_ms"].values
        * (1 + topo_gain * severity * (0.20 + 0.10 * tl + 0.10 * rl) + rng.normal(0, 0.10 * severity, len(out))))

    if "sched_duration_ms" in out.columns:
        out["sched_duration_ms"] = np.maximum(1.0, out["sched_duration_ms"].values
            * (1 + topo_gain * severity * (0.18 + 0.10 * dl) + rng.normal(0, 0.08 * severity, len(out))))
    if "idle_variance" in out.columns:
        out["idle_variance"] = np.maximum(0.0, out["idle_variance"].values
            + topo_gain * severity * (0.10 * rl + 0.08 * dl) + rng.normal(0, 0.10 * severity, len(out)))

    ld = np.maximum(out["logical_depth"].values, 1.0)
    lq = np.maximum(out["logical_twoq"].values,  1.0)
    routed_twoq = out["extra_twoq"].values + out["logical_twoq"].values

    out["depth_overhead"] = out["routed_depth"].values / ld
    out["twoq_overhead"]  = routed_twoq / lq
    out["swap_fraction"]  = np.clip(out["swap_equiv"].values / np.maximum(routed_twoq, 1.0), 0.0, 1.0)
    if "logical_total_ops" in out.columns:
        approx_total = np.maximum(out["logical_total_ops"].values + out["extra_twoq"].values
                                  + out["extra_depth"].values, 1.0)
        out["cx_fraction"] = np.clip(routed_twoq / approx_total, 0.0, 1.0)

    return out


# ════════════════════════════════════════════════════════════════════════════
# 5.  EXPERIMENT CONFIG
# ════════════════════════════════════════════════════════════════════════════

TOPOLOGY_FAMILY  = "ladder"
NUM_QUBITS       = 8
RESOLUTIONS      = [4, 8, 12]
N_SAMPLES        = 20
N_STEPS          = 3
BASE_SEED        = 42
DRIFT_SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.00]

PROBE_FEATURES = [
    "swap_equiv", "extra_twoq", "routed_depth", "extra_depth",
    "transpile_ms", "depth_overhead", "twoq_overhead",
    "swap_fraction", "sched_duration_ms", "idle_variance",
]
RATIO_FEATURES = ["swap_fraction", "cx_fraction"]


# ════════════════════════════════════════════════════════════════════════════
# 6.  METRIC HELPERS
# ════════════════════════════════════════════════════════════════════════════

def mean_rel_dev(base: pd.DataFrame, drifted: pd.DataFrame, col: str) -> float:
    if col not in base.columns or col not in drifted.columns:
        return float("nan")
    b = base[col].values.astype(float)
    d = drifted[col].values.astype(float)
    denom = np.where(np.abs(b) > 1e-9, np.abs(b), 1.0)
    return float(np.mean(np.abs(d - b) / denom))


def ratio_violation_rate(df: pd.DataFrame) -> float:
    cols = [c for c in RATIO_FEATURES if c in df.columns]
    if not cols:
        return 0.0
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        mask |= (df[c].values < 0) | (df[c].values > 1)
    return float(mask.mean())


def primitive_nonneg_rate(df: pd.DataFrame) -> float:
    cols = [c for c in ["swap_equiv", "extra_twoq", "routed_depth", "extra_depth"]
            if c in df.columns]
    ok = np.ones(len(df), dtype=bool)
    for c in cols:
        ok &= df[c].values >= 0
    return float(ok.mean())


# ════════════════════════════════════════════════════════════════════════════
# 7.  RUN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def run_experiment() -> pd.DataFrame:
    print("=" * 64)
    print("EXP5 — Hardware Drift Validation  (Qiskit 2.x)")
    print("=" * 64)

    print("\n[1/4] Building boundary baseline …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bnd_ds = build_boundary_dataset(
            topology_family=TOPOLOGY_FAMILY, num_qubits=NUM_QUBITS,
            n_samples_per_class=N_SAMPLES, n_steps=N_STEPS, seed=BASE_SEED,
        )
    print(f"      {len(bnd_ds.df)} rows, {bnd_ds.df.shape[1]} cols")

    print("[2/4] Building scale baseline …")
    scl_ds = build_scale_dataset(
        resolutions=RESOLUTIONS, n_samples_per_res=N_SAMPLES,
        topology_families=[TOPOLOGY_FAMILY], template_families=["qft"],
        n_steps=N_STEPS, seed=BASE_SEED + 1,
    )
    print(f"      {len(scl_ds.df)} rows, {scl_ds.df.shape[1]} cols")

    print("[3/4] Augmenting operational features …")
    b_base = augment_operational_features(bnd_ds.df, seed=BASE_SEED + 10)
    s_base = augment_operational_features(scl_ds.df, seed=BASE_SEED + 11)

    print("[4/4] Sweeping drift severities …")
    results = []
    for sev in DRIFT_SEVERITIES:
        for ds_name, base_df in [("boundary", b_base), ("scale", s_base)]:
            drifted = apply_hardware_drift(base_df, sev, TOPOLOGY_FAMILY, BASE_SEED + int(sev * 100))
            row = {
                "dataset": ds_name, "severity": sev,
                "ratio_viol":  ratio_violation_rate(drifted),
                "prim_nonneg": primitive_nonneg_rate(drifted),
            }
            for feat in PROBE_FEATURES:
                row[f"rd_{feat}"] = mean_rel_dev(base_df, drifted, feat)
            results.append(row)

    df = pd.DataFrame(results)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 130)
    print("\n── Stability summary ──────────────────────────────────────────────────────")
    key_cols = ["dataset", "severity", "ratio_viol", "prim_nonneg",
                "rd_routed_depth", "rd_swap_equiv", "rd_swap_fraction",
                "rd_depth_overhead", "rd_sched_duration_ms"]
    print(df[[c for c in key_cols if c in df.columns]].to_string(index=False))

    viol = df[df["ratio_viol"] > 0]
    print("\n✓ No ratio violations." if viol.empty else f"\n⚠ Ratio violations: {len(viol)} conditions")
    print("✓ All primitives non-negative." if (df["prim_nonneg"] == 1.0).all() else "⚠ Non-negativity failures.")
    print("\n✓ exp5 complete.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 8.  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

COLORS   = {"boundary": "#E05C5C", "scale": "#4A90D9"}
MARKERS  = {"boundary": "o",       "scale": "s"}

PANEL_FEATURES = [
    ("rd_routed_depth",      "Routed Depth",           "Mean Relative Deviation"),
    ("rd_swap_equiv",        "SWAP Equivalents",        "Mean Relative Deviation"),
    ("rd_extra_twoq",        "Extra 2Q Gates",          "Mean Relative Deviation"),
    ("rd_transpile_ms",      "Transpile Time (ms)",     "Mean Relative Deviation"),
    ("rd_sched_duration_ms", "Sched Duration (ms)",     "Mean Relative Deviation"),
    ("rd_swap_fraction",     "Swap Fraction",           "Mean Relative Deviation"),
    ("rd_depth_overhead",    "Depth Overhead",          "Mean Relative Deviation"),
    ("rd_idle_variance",     "Idle Variance",           "Mean Relative Deviation"),
]

def plot_results(df: pd.DataFrame, out_path: str = "exp5_results.png") -> None:
    sev  = sorted(df["severity"].unique())
    bnd  = df[df["dataset"] == "boundary"].sort_values("severity")
    scl  = df[df["dataset"] == "scale"].sort_values("severity")

    fig  = plt.figure(figsize=(18, 14), facecolor="#0F1117")
    fig.patch.set_facecolor("#0F1117")

    # ── title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "EXP5 — Hardware Drift Validation",
             ha="center", va="top", fontsize=18, fontweight="bold",
             color="white", fontfamily="monospace")
    fig.text(0.5, 0.945,
             f"Topology: {TOPOLOGY_FAMILY} | Boundary: {NUM_QUBITS}Q | "
             f"Scale: {RESOLUTIONS}Q | {N_SAMPLES} samples/class | QFT template",
             ha="center", va="top", fontsize=10, color="#AAAAAA", fontfamily="monospace")

    gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                                  top=0.92, bottom=0.06, hspace=0.42,
                                  left=0.06, right=0.97)

    # ── Panel row A: 4 feature deviation plots ───────────────────────────────
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.38)
    for i, (col, title, ylabel) in enumerate(PANEL_FEATURES[:4]):
        ax = fig.add_subplot(gs_a[i])
        _style_ax(ax)
        if col in bnd.columns:
            ax.plot(bnd["severity"], bnd[col], color=COLORS["boundary"],
                    marker=MARKERS["boundary"], lw=2, ms=6, label="boundary")
        if col in scl.columns:
            ax.plot(scl["severity"], scl[col], color=COLORS["scale"],
                    marker=MARKERS["scale"],    lw=2, ms=6, label="scale", linestyle="--")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Drift Severity", color="#AAAAAA", fontsize=8)
        if i == 0:
            ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=8)
        ax.set_xticks(sev)
        ax.tick_params(colors="#AAAAAA", labelsize=8)

    # ── Panel row B: next 4 feature deviation plots ──────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.38)
    for i, (col, title, ylabel) in enumerate(PANEL_FEATURES[4:]):
        ax = fig.add_subplot(gs_b[i])
        _style_ax(ax)
        if col in bnd.columns:
            ax.plot(bnd["severity"], bnd[col], color=COLORS["boundary"],
                    marker=MARKERS["boundary"], lw=2, ms=6, label="boundary")
        if col in scl.columns:
            ax.plot(scl["severity"], scl[col], color=COLORS["scale"],
                    marker=MARKERS["scale"],    lw=2, ms=6, label="scale", linestyle="--")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Drift Severity", color="#AAAAAA", fontsize=8)
        if i == 0:
            ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=8)
        ax.set_xticks(sev)
        ax.tick_params(colors="#AAAAAA", labelsize=8)

    # ── Panel row C: summary heatmap + constraint health bar ─────────────────
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[2],
                                             wspace=0.40, width_ratios=[2.2, 1])

    # Heatmap: rel-dev for selected features × severity, split boundary / scale
    heat_cols = ["rd_routed_depth", "rd_swap_equiv", "rd_extra_twoq",
                 "rd_transpile_ms", "rd_sched_duration_ms",
                 "rd_swap_fraction", "rd_depth_overhead", "rd_idle_variance"]
    heat_labels = ["Routed Depth", "SWAP Equiv", "Extra 2Q",
                   "Transpile ms", "Sched ms",
                   "Swap Frac", "Depth OH", "Idle Var"]

    ax_heat = fig.add_subplot(gs_c[0])
    _style_ax(ax_heat)

    # Build matrix: rows = (dataset, severity), cols = features
    datasets_order = ["boundary", "scale"]
    mat = np.zeros((len(sev) * 2, len(heat_cols)))
    row_labels = []
    for di, dsn in enumerate(datasets_order):
        sub = df[df["dataset"] == dsn].sort_values("severity")
        for si, sv in enumerate(sev):
            ridx = di * len(sev) + si
            row_labels.append(f"{dsn[:3]}  s={sv:.2f}")
            for ci, c in enumerate(heat_cols):
                mat[ridx, ci] = sub[sub["severity"] == sv][c].values[0] if c in sub.columns else np.nan

    im = ax_heat.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.35)
    ax_heat.set_xticks(range(len(heat_cols)))
    ax_heat.set_xticklabels(heat_labels, rotation=35, ha="right", color="#CCCCCC", fontsize=7)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, color="#CCCCCC", fontsize=7)
    ax_heat.set_title("Feature Deviation Heatmap\n(rel-dev from baseline)", color="white", fontsize=9, pad=6)

    # colour-code row labels
    for ti, lbl in enumerate(row_labels):
        dsn = "boundary" if ti < len(sev) else "scale"
        ax_heat.get_yticklabels()[ti].set_color(COLORS[dsn])

    # annotate cells
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax_heat.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                             fontsize=6, color="black" if v > 0.15 else "white")

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=7)
    cbar.set_label("Rel. Dev.", color="#AAAAAA", fontsize=7)

    # Constraint health bar chart
    ax_bar = fig.add_subplot(gs_c[1])
    _style_ax(ax_bar)

    bar_x    = np.array(sev)
    bar_w    = 0.10
    for di, dsn in enumerate(datasets_order):
        sub = df[df["dataset"] == dsn].sort_values("severity")
        offset = (di - 0.5) * bar_w
        vals   = (1.0 - sub["ratio_viol"].values) * 100   # % healthy
        bars   = ax_bar.bar(bar_x + offset, vals, width=bar_w,
                             color=COLORS[dsn], alpha=0.85, label=dsn)

    ax_bar.set_ylim(0, 110)
    ax_bar.axhline(100, color="#44FF88", lw=1.2, linestyle="--", alpha=0.7)
    ax_bar.set_xlabel("Drift Severity", color="#AAAAAA", fontsize=8)
    ax_bar.set_ylabel("% Rows Constraint-Healthy", color="#AAAAAA", fontsize=8)
    ax_bar.set_title("Ratio Constraint Health\n(100% = no violations)", color="white", fontsize=9, pad=6)
    ax_bar.set_xticks(sev)
    ax_bar.tick_params(colors="#AAAAAA", labelsize=8)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=COLORS["boundary"], marker="o", lw=2, ms=6, label="Boundary dataset"),
        Line2D([0], [0], color=COLORS["scale"],    marker="s", lw=2, ms=6, linestyle="--", label="Scale dataset"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               facecolor="#1A1D27", edgecolor="#444", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out_path}")


def _style_ax(ax):
    ax.set_facecolor("#1A1D27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(True, color="#2A2D3A", linewidth=0.7, linestyle="--")
    ax.tick_params(colors="#AAAAAA")


# ════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_df = run_experiment()
    plot_results(results_df, out_path="exp5_results.png")
