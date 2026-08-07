"""
Group statistical plotting and reporting.

This stage consumes group-statistics derivatives from `batch_group_statistics`
and writes summary tables plus selected label time-course figures. Optional
brain-behavior correlations use Stage 1 behavior TSV derivatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meg_tokens.analysis.statistics import get_significance_windows
from meg_tokens.core import normalize_subject_id
from meg_tokens.io import (
    DerivativeLayout,
    derivative_path,
    ensure_dir,
    load_array,
    require_file,
    save_sidecar,
    save_table,
)
from meg_tokens.reports.meg import plot_correlation
from meg_tokens.behavior.tdms import started_trials


def stats_derivative_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    suffix: str,
    extension: str,
) -> Path:
    return DerivativeLayout(output_root).group_stats(
        conditions=conditions,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        suffix=suffix,
        extension=extension,
    )


def figure_derivative_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    label_index: int,
    component_index: Optional[int] = None,
    suffix: str = "stattimecourse",
    extension: str = ".png",
) -> Path:
    desc_parts = [
        conditions[0].lower(),
        "vs",
        conditions[1].lower(),
        align_to,
        source_method,
        parc,
    ]
    if component_index is not None:
        desc_parts.append(f"component{component_index:02d}")
    desc_parts.append(f"label{label_index:03d}")
    return derivative_path(
        output_root,
        subject="group",
        datatype="fig",
        task="tokens",
        description="-".join(desc_parts),
        suffix=suffix,
        extension=extension,
    )


def find_stats_array(
    stats_dir: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    suffix: str,
) -> Path:
    """Return the expected Stage 6 stats array/table path."""
    extension = ".tsv" if suffix == "sigwindows" else ".npy"
    path = stats_derivative_path(
        stats_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix=suffix,
        extension=extension,
    )
    return require_file(path, purpose=f"group statistics {suffix}")


def load_group_stats(
    stats_dir: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
) -> dict[str, object]:
    """Load and validate group statistics derivatives."""
    tstat_path = find_stats_array(
        stats_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix="tstat",
    )
    pvalue_path = find_stats_array(
        stats_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix="pvalue",
    )
    contrast_path = find_stats_array(
        stats_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix="contrast",
    )

    tstat = load_array(tstat_path, require_sidecar=True)
    pvalue = load_array(pvalue_path, require_sidecar=True)
    contrast = load_array(contrast_path, require_sidecar=True)

    dims = tuple(tstat.metadata.get("dims", []))
    if dims not in {("label", "time"), ("component", "label", "time")}:
        raise ValueError(f"Unsupported stats dimensions in {tstat_path}: {dims}")
    if tuple(pvalue.metadata.get("dims", [])) != dims:
        raise ValueError("tstat and pvalue dimensions do not match")
    if tuple(contrast.metadata.get("dims", [])) != ("subject", *dims):
        raise ValueError("contrast dimensions must be subject plus tstat dimensions")
    if tstat.data.shape != pvalue.data.shape:
        raise ValueError("tstat and pvalue shapes do not match")
    if contrast.data.shape[1:] != tstat.data.shape:
        raise ValueError("contrast feature shape does not match tstat shape")

    coords = tstat.metadata.get("coords", {})
    return {
        "tstat": tstat.data,
        "pvalue": pvalue.data,
        "contrast": contrast.data,
        "dims": dims,
        "coords": coords,
        "subjects": contrast.metadata.get("coords", {}).get("subject", []),
        "metadata": tstat.metadata.get("metadata", {}),
        "paths": {
            "tstat": str(tstat_path),
            "pvalue": str(pvalue_path),
            "contrast": str(contrast_path),
        },
    }


def _as_float_array(values, fallback_length: int) -> np.ndarray:
    if values is None:
        return np.arange(fallback_length, dtype=float)
    return np.asarray(values, dtype=float)


def _series_for_unit(array: np.ndarray, dims: Sequence[str], unit: dict[str, object]) -> np.ndarray:
    if tuple(dims) == ("label", "time"):
        return array[int(unit["label_index"])]
    return array[int(unit["component_index"]), int(unit["label_index"])]


def _contrast_for_unit(contrast: np.ndarray, dims: Sequence[str], unit: dict[str, object]) -> np.ndarray:
    if tuple(dims) == ("label", "time"):
        return contrast[:, int(unit["label_index"]), :]
    return contrast[:, int(unit["component_index"]), int(unit["label_index"]), :]


def _unit_score(tstat: np.ndarray, dims: Sequence[str], unit: dict[str, object]) -> float:
    series = _series_for_unit(tstat, dims, unit)
    if not np.any(np.isfinite(series)):
        return 0.0
    return float(np.nanmax(np.abs(series)))


def select_plot_units(
    tstat: np.ndarray,
    dims: Sequence[str],
    coords: dict[str, object],
    *,
    labels: Optional[Sequence[str]] = None,
    top_n: int = 8,
) -> list[dict[str, object]]:
    """Select label or component-label units to plot."""
    label_names = list(coords.get("label", [f"label_{idx}" for idx in range(tstat.shape[-2])]))
    label_lookup = {str(label): idx for idx, label in enumerate(label_names)}

    units = []
    if tuple(dims) == ("label", "time"):
        for label_idx, label_name in enumerate(label_names):
            units.append({"label_index": label_idx, "label": label_name})
    else:
        components = list(coords.get("component", [str(idx) for idx in range(tstat.shape[0])]))
        for comp_idx, component in enumerate(components):
            for label_idx, label_name in enumerate(label_names):
                units.append({
                    "component_index": comp_idx,
                    "component": component,
                    "label_index": label_idx,
                    "label": label_name,
                })

    if labels:
        selected = []
        for item in labels:
            if str(item).isdigit():
                label_idx = int(item)
            else:
                if item not in label_lookup:
                    raise ValueError(f"Requested label is absent from stats coordinates: {item}")
                label_idx = label_lookup[item]
            selected.extend([unit for unit in units if int(unit["label_index"]) == label_idx])
        return selected

    ranked = sorted(units, key=lambda unit: _unit_score(tstat, dims, unit), reverse=True)
    return ranked[:max(1, int(top_n))]


def _peak_index(series: np.ndarray, peak_mode: str) -> int:
    finite = np.isfinite(series)
    if not np.any(finite):
        return -1
    candidate = np.where(finite)[0]
    values = series[finite]
    if peak_mode == "min":
        return int(candidate[np.argmin(values)])
    if peak_mode == "max":
        return int(candidate[np.argmax(values)])
    if peak_mode == "abs":
        return int(candidate[np.argmax(np.abs(values))])
    raise ValueError("peak_mode must be one of: min, max, abs")


def build_stats_summary(
    tstat: np.ndarray,
    pvalue: np.ndarray,
    dims: Sequence[str],
    coords: dict[str, object],
    *,
    alpha: float,
    peak_mode: str,
) -> pd.DataFrame:
    """Build a per-label summary table for the full stats result."""
    units = select_plot_units(tstat, dims, coords, top_n=np.prod(tstat.shape[:-1]))
    times = _as_float_array(coords.get("time_sec"), tstat.shape[-1])
    rows = []

    for unit in units:
        t_series = _series_for_unit(tstat, dims, unit)
        p_series = _series_for_unit(pvalue, dims, unit)
        finite_p = np.isfinite(p_series)
        sig = finite_p & (p_series <= alpha)
        onset_idx = int(np.where(sig)[0][0]) if np.any(sig) else -1
        peak_idx = _peak_index(t_series, peak_mode)
        windows = get_significance_windows(np.where(finite_p, p_series, np.inf), alpha=alpha)
        row = {
            "label": unit["label"],
            "label_index": int(unit["label_index"]),
            "onset_index": onset_idx,
            "onset_time_sec": times[onset_idx] if onset_idx >= 0 else np.nan,
            "peak_index": peak_idx,
            "peak_time_sec": times[peak_idx] if peak_idx >= 0 else np.nan,
            "peak_t": t_series[peak_idx] if peak_idx >= 0 else np.nan,
            "max_abs_t": float(np.nanmax(np.abs(t_series))) if np.any(np.isfinite(t_series)) else np.nan,
            "min_p": float(np.nanmin(p_series)) if np.any(finite_p) else np.nan,
            "n_significant_timepoints": int(np.sum(sig)),
            "n_significant_windows": len(windows),
            "alpha": float(alpha),
            "peak_mode": peak_mode,
        }
        if "component_index" in unit:
            row["component"] = unit["component"]
            row["component_index"] = int(unit["component_index"])
        rows.append(row)

    return pd.DataFrame(rows)


def save_timecourse_figure(
    path: Path,
    *,
    times: np.ndarray,
    subject_series: np.ndarray,
    t_series: np.ndarray,
    p_series: np.ndarray,
    unit: dict[str, object],
    conditions: Sequence[str],
    alpha: float,
) -> None:
    """Save a subject-contrast time-course plot for one unit."""
    ensure_dir(path.parent)
    mean = np.nanmean(subject_series, axis=0)
    n_finite = np.sum(np.isfinite(subject_series), axis=0)
    sem = np.nanstd(subject_series, axis=0, ddof=1) / np.sqrt(np.maximum(n_finite, 1))
    windows = get_significance_windows(np.where(np.isfinite(p_series), p_series, np.inf), alpha=alpha)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(times, mean, color="#2f5d62", lw=2, label=f"{conditions[0]} - {conditions[1]}")
    ax.fill_between(times, mean - sem, mean + sem, color="#2f5d62", alpha=0.22, linewidth=0)
    ax.axhline(0.0, color="#333333", lw=1, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subject contrast")
    title = str(unit["label"])
    if "component" in unit:
        title = f"{title} | component {unit['component']}"
    ax.set_title(title)

    for idx, (start, end) in enumerate(windows):
        start_time = times[start]
        end_time = times[end - 1]
        ax.axvspan(start_time, end_time, color="#b85c38", alpha=0.18, label="p <= alpha" if idx == 0 else None)

    ax2 = ax.twinx()
    ax2.plot(times, t_series, color="#6f4e7c", lw=1.3, alpha=0.72, label="t statistic")
    ax2.set_ylabel("t statistic")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _behavior_tables_for_subject(behavior_dir: str, subject: str, conditions: Sequence[str]) -> list[Path]:
    subject = normalize_subject_id(subject)
    beh_dir = Path(behavior_dir) / "derivatives" / "meg-tokens" / f"sub-{subject}" / "beh"
    paths = []
    for condition in conditions:
        paths.extend(beh_dir.glob(f"sub-{subject}_task-tokens_run-*_desc-{condition.lower()}_beh.tsv"))
    return sorted(path for path in paths if path.is_file())


def mean_raw_rt_by_subject(behavior_dir: str, subjects: Sequence[str], conditions: Sequence[str]) -> dict[str, float]:
    """Average real Stage 1 rawRT values per subject for the requested conditions."""
    values = {}
    for subject in subjects:
        subject = normalize_subject_id(subject)
        rt_values = []
        for path in _behavior_tables_for_subject(behavior_dir, subject, conditions):
            table = started_trials(pd.read_csv(path, sep="\t"))
            if "rawRT" in table.columns:
                series = table["rawRT"]
            else:
                required = {"tEnterTarget", "tGO", "nChoiceMade"}
                missing = required - set(table.columns)
                if missing:
                    raise ValueError(f"Behavior table {path} is missing RT columns: {sorted(missing)}")
                series = (table["tEnterTarget"] - table["tGO"]).where(table["nChoiceMade"] > 0)
            rt_values.extend(pd.to_numeric(series, errors="coerce").dropna().tolist())
        if rt_values:
            values[subject] = float(np.mean(rt_values))
    return values


def save_behavior_correlations(
    output_root: str,
    *,
    behavior_dir: str,
    subjects: Sequence[str],
    contrast: np.ndarray,
    dims: Sequence[str],
    coords: dict[str, object],
    units: Sequence[dict[str, object]],
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    peak_mode: str,
) -> Optional[Path]:
    """Save optional brain-behavior correlation table and figures."""
    mean_rt = mean_raw_rt_by_subject(behavior_dir, subjects, conditions)
    common_subjects = [normalize_subject_id(subject) for subject in subjects if normalize_subject_id(subject) in mean_rt]
    if len(common_subjects) < 3:
        raise ValueError("At least three subjects with behavior tables are required for correlations")

    subject_index = {normalize_subject_id(subject): idx for idx, subject in enumerate(subjects)}
    times = _as_float_array(coords.get("time_sec"), contrast.shape[-1])
    rows = []

    for unit in units:
        subject_series = _contrast_for_unit(contrast, dims, unit)
        rt_values = []
        peak_times = []
        for subject in common_subjects:
            series = subject_series[subject_index[subject]]
            peak_idx = _peak_index(series, peak_mode)
            if peak_idx < 0:
                continue
            rt_values.append(mean_rt[subject])
            peak_times.append(float(times[peak_idx]))
        if len(rt_values) < 3:
            continue
        r_value = float(np.corrcoef(rt_values, peak_times)[0, 1])
        rows.append({
            "label": unit["label"],
            "label_index": int(unit["label_index"]),
            "component": unit.get("component"),
            "component_index": unit.get("component_index"),
            "n_subjects": len(rt_values),
            "pearson_r": r_value,
            "peak_mode": peak_mode,
        })

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        plot_correlation(
            np.asarray(rt_values),
            np.asarray(peak_times),
            x_label="Mean rawRT (ms)",
            y_label="Subject contrast peak time (s)",
            title=f"{unit['label']}: {conditions[0]} vs {conditions[1]}",
            ax=ax,
        )
        fig_path = figure_derivative_path(
            output_root,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            label_index=int(unit["label_index"]),
            component_index=unit.get("component_index"),
            suffix="statcorrelation",
        )
        ensure_dir(fig_path.parent)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        save_sidecar(fig_path, {
            "stage": "statistical_plotting",
            "kind": "brain_behavior_correlation",
            "conditions": list(conditions),
            "label": unit["label"],
            "label_index": int(unit["label_index"]),
            "component": unit.get("component"),
            "component_index": unit.get("component_index"),
            "subjects": common_subjects,
            "behavior_metric": "mean_rawRT",
            "peak_mode": peak_mode,
            "pearson_r": r_value,
        })

    table_path = stats_derivative_path(
        output_root,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix="statcorrelations",
        extension=".tsv",
    )
    save_table(
        table_path,
        pd.DataFrame(rows),
        metadata={
            "stage": "statistical_plotting",
            "kind": "brain_behavior_correlation_table",
            "conditions": list(conditions),
            "subjects": common_subjects,
            "behavior_metric": "mean_rawRT",
            "peak_mode": peak_mode,
        },
    )
    return table_path


def run_group_statistics_plotting(
    stats_dir: str,
    output_dir: str,
    conditions: Sequence[str] = ("Fast", "Slow"),
    *,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    labels: Optional[Sequence[str]] = None,
    top_n: int = 8,
    alpha: float = 0.05,
    peak_mode: str = "min",
    behavior_dir: Optional[str] = None,
    correlate_behavior: bool = False,
) -> dict[str, object]:
    """Generate summary tables and selected plots from group stats derivatives."""
    if len(conditions) != 2:
        raise ValueError("Statistical plotting currently requires exactly two conditions")
    conditions = [str(condition) for condition in conditions]
    bundle = load_group_stats(
        stats_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
    )
    tstat = bundle["tstat"]
    pvalue = bundle["pvalue"]
    contrast = bundle["contrast"]
    dims = bundle["dims"]
    coords = bundle["coords"]
    subjects = bundle["subjects"]

    summary = build_stats_summary(
        tstat,
        pvalue,
        dims,
        coords,
        alpha=alpha,
        peak_mode=peak_mode,
    )
    summary_path = stats_derivative_path(
        output_dir,
        conditions=conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        suffix="statsummary",
        extension=".tsv",
    )
    save_table(
        summary_path,
        summary,
        metadata={
            "stage": "statistical_plotting",
            "kind": "stats_summary",
            "conditions": conditions,
            "alignment": align_to,
            "source_method": source_method,
            "parcellation": parc,
            "alpha": float(alpha),
            "peak_mode": peak_mode,
            "inputs": bundle["paths"],
        },
    )

    selected_units = select_plot_units(tstat, dims, coords, labels=labels, top_n=top_n)
    times = _as_float_array(coords.get("time_sec"), tstat.shape[-1])
    figure_paths = []
    for unit in selected_units:
        figure_path = figure_derivative_path(
            output_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            label_index=int(unit["label_index"]),
            component_index=unit.get("component_index"),
            suffix="stattimecourse",
        )
        save_timecourse_figure(
            figure_path,
            times=times,
            subject_series=_contrast_for_unit(contrast, dims, unit),
            t_series=_series_for_unit(tstat, dims, unit),
            p_series=_series_for_unit(pvalue, dims, unit),
            unit=unit,
            conditions=conditions,
            alpha=alpha,
        )
        save_sidecar(
            figure_path,
            {
                "stage": "statistical_plotting",
                "kind": "stat_timecourse_figure",
                "conditions": conditions,
                "alignment": align_to,
                "source_method": source_method,
                "parcellation": parc,
                "label": unit["label"],
                "label_index": int(unit["label_index"]),
                "component": unit.get("component"),
                "component_index": unit.get("component_index"),
                "alpha": float(alpha),
                "inputs": bundle["paths"],
            },
        )
        figure_paths.append(figure_path)

    correlation_path = None
    if correlate_behavior:
        if behavior_dir is None:
            raise ValueError("--behavior_dir is required when --correlate_behavior is used")
        correlation_path = save_behavior_correlations(
            output_dir,
            behavior_dir=behavior_dir,
            subjects=subjects,
            contrast=contrast,
            dims=dims,
            coords=coords,
            units=selected_units,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            peak_mode=peak_mode,
        )

    print(f"Saved stats summary: {summary_path}")
    print(f"Saved {len(figure_paths)} stat time-course figures")
    return {
        "summary": summary_path,
        "figures": figure_paths,
        "correlations": correlation_path,
    }
