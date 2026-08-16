"""Orchestrate the behavioral figure battery.

Reads Stage 2/2b **group** derivatives (via `BehaviorTableSet`). A missing
derivative raises `FileNotFoundError` naming the path and the command that
writes it (`meg-tokens behavior characterization`, or `meg-tokens behavior
ssm-fit` for any `ssm*` table), unless `skip_missing=True`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from meg_tokens.io import DerivativeLayout, save_sidecar
from meg_tokens.reports import style
from meg_tokens.reports.annotations import SIGNIFICANCE_CONVENTION
from meg_tokens.reports.behavior import (
    FigureSpec,
    REGISTRY,
    list_behavior_figures,
    resolve_figure_keys,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet

__all__ = ["list_behavior_figures", "run_behavior_plotting"]

_REGISTRY_BY_KEY = {spec.key: spec for spec in REGISTRY}


def _render_one(
    spec: FigureSpec,
    tables: BehaviorTableSet,
    output_layout: DerivativeLayout,
    formats: Sequence[str],
) -> Path:
    figure, extra_metadata = spec.builder(tables)
    try:
        base_path = output_layout.path(
            subject="group",
            datatype="fig",
            description=f"{spec.analysis}-{spec.view}",
            suffix="behavior",
            extension=formats[0],
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for extension in formats:
            figure_path = base_path.with_suffix(extension)
            # apply_publication_style()'s rc_context has already exited by
            # the time this runs (the builder returns the figure, it doesn't
            # save it), so savefig.dpi=400 from that context is gone -- pass
            # it explicitly or every raster figure silently saves at
            # matplotlib's 100 dpi default.
            figure.savefig(figure_path, dpi=style.SAVEFIG_DPI)
            written.append(figure_path)

        # tables is shared across every figure in this run (for caching), so
        # tables.sources accumulates across figures -- it is NOT a per-figure
        # source list. Each builder's own columns_read names exactly what it
        # read, so use that to select this figure's own sources.
        read_names = extra_metadata.get("columns_read", {}).keys()
        source_derivatives = [
            tables.sources[name] for name in read_names if name in tables.sources
        ]

        metadata = {
            "stage": "behavior_report",
            "figure": spec.key,
            "analysis": spec.analysis,
            "view": spec.view,
            "significance_convention": SIGNIFICANCE_CONVENTION,
            "multiplicity_correction": "none",
            "caveat": spec.caveat,
            "formats": list(formats),
            "dpi": 400,
            "source_derivatives": source_derivatives,
            **extra_metadata,
        }
        # io.sidecar_path collapses every format of the same figure onto one
        # .json (Path.with_suffix); write it once, after all formats.
        save_sidecar(written[0], metadata)
        return written[0]
    finally:
        plt.close(figure)


def run_behavior_plotting(
    behavior_dir: str,
    output_figures_dir: str,
    subjects_list: list[str] | None = None,
    neural_metrics_csv: str | None = None,
    *,
    figures: Sequence[str] = ("all",),
    formats: Sequence[str] = (".pdf", ".png"),
    skip_missing: bool = False,
) -> list[Path]:
    """Render the selected behavioral figures and return every file written."""
    input_layout = DerivativeLayout(behavior_dir)
    output_layout = DerivativeLayout(output_figures_dir)
    tables = BehaviorTableSet(
        layout=input_layout,
        subjects=tuple(subjects_list) if subjects_list else None,
        neural_metrics_path=neural_metrics_csv,
    )

    keys = resolve_figure_keys(tuple(figures))
    outputs: list[Path] = []
    for key in keys:
        spec = _REGISTRY_BY_KEY[key]
        try:
            outputs.append(_render_one(spec, tables, output_layout, tuple(formats)))
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
    return outputs
