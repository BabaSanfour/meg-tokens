"""The band choice from spectral peaks, drawn (``workflows/spectral.py``, ``band_choice``).

Two figures per stream: specparam's own group report of the model, the layout
of the preprint's Fig S1; and the band choice on one frequency axis, the
spectrum above its aperiodic fit, the peak frequencies, the bands and how they
move with the line, beside the fit's quality. One stream per figure, in ink:
colour carries no identity there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import matplotlib.ticker
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from meg_tokens.core import BandChoiceConfig, ProjectConfig, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.reports import style

BAND_SHADE = "#f0efec"
FORMATS = (".pdf", ".png")


def plot_band_choice(path: str | Path) -> Figure:
    """One stream's band choice: A the power above the aperiodic fit, with the bands above the mean bin and
    their edges' 95 % intervals beneath it; B each subject's peaks per bin and the mean bin; C the bands as
    the line moves from 0.5 to 2 times the mean bin; D the aperiodic exponents and E the fits' R2."""
    with h5py.File(path, "r") as handle:
        data = {name: handle[name][:] for name in handle}
    names = [name.decode() for name in data["band_names"]]
    edges, freqs = data["edges"], data["freqs"]
    symbols = {"theta": "θ", "alpha": "α", "beta": "β"}

    with style.apply_publication_style():
        fig = Figure(figsize=(style.WIDTH_DOUBLE_IN, 5.6), layout="constrained")
        grid = fig.add_gridspec(4, 2, height_ratios=[2.3, 0.35, 2.0, 0.9], width_ratios=[2.2, 1.0])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_bars = fig.add_subplot(grid[1, 0], sharex=ax_a)
        ax_b = fig.add_subplot(grid[2, 0], sharex=ax_a)
        ax_c = fig.add_subplot(grid[3, 0], sharex=ax_a)
        right = grid[:, 1].subgridspec(2, 1)
        ax_d, ax_e = fig.add_subplot(right[0]), fig.add_subplot(right[1])
        for ax in (ax_a, ax_bars, ax_b, ax_c):
            for lo, hi in data["preprint_bands"]:
                ax.axvspan(lo, hi, color=BAND_SHADE, lw=0, zorder=0)

        ax_a.fill_between(freqs, *data["periodic_interval"], color=style.INK_MUTED, alpha=0.35, lw=0)
        ax_a.plot(freqs, data["periodic"].mean(axis=0), color=style.INK)
        ax_a.set_ylabel("Power above\naperiodic fit (log$_{10}$)")
        top = ax_a.secondary_xaxis("top")
        top.set_xticks(data["preprint_bands"].mean(axis=1), [symbols.get(name, name) for name in names])
        top.tick_params(length=0)
        top.spines["top"].set_visible(False)

        for name, (lo, hi), resampled in zip(names, data["bands"], data["bands_resampled"]):
            if np.isnan(lo):
                continue
            ax_bars.plot([lo, hi], [0.5, 0.5], color=style.INK, lw=3.5, solid_capstyle="butt")
            for column in (0, 1):
                a, b = np.nanpercentile(resampled[:, column], [2.5, 97.5])
                ax_bars.plot([a, b], [0.5, 0.5], color=style.INK, lw=0.9)
                ax_bars.plot([a, a, np.nan, b, b], [0.3, 0.7, np.nan, 0.3, 0.7], color=style.INK, lw=0.9)
        ax_bars.set_ylim(0, 1)
        ax_bars.set_yticks([])
        for side in ("left", "bottom"):
            ax_bars.spines[side].set_visible(False)

        mean = data["shares"].mean(axis=0)
        low, high = data["shares_interval"]
        ax_b.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post", color=style.INK_MUTED,
                          alpha=0.35, lw=0)
        ax_b.step(edges, np.r_[mean, mean[-1]], where="post", color=style.INK)
        ax_b.axhline(mean.mean(), color=style.INK, lw=0.8, ls=(0, (3, 2)))
        ax_b.set_ylabel("Peaks per\nsubject (%)")
        ax_b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))

        lines = data["lines"]
        step = (lines[1] - lines[0]) / 2
        ax_c.pcolormesh(edges, np.r_[lines - step, lines[-1] + step], np.ma.masked_where(~data["sweep"], data["sweep"]),
                        cmap=ListedColormap([style.INK_SECONDARY]), shading="flat", rasterized=True)
        ax_c.axhline(1.0, color=style.INK, lw=0.8, ls=(0, (3, 2)))
        ax_c.set_ylim(lines[0], lines[-1])
        ax_c.set_yticks([0.5, 1, 2])
        ax_c.set_ylabel("Line\n(× mean)")
        ax_c.set_xlabel("Frequency (Hz)")
        ax_a.set_xlim(edges[0], edges[-1])
        for ax in (ax_a, ax_bars, ax_b):
            ax.tick_params(labelbottom=False)

        for ax, values, mark, label in ((ax_d, data["exponent"], 0.0, "Exponent"),
                                        (ax_e, data["r2"], 0.9, "R$^2$ of the fit")):
            ordered = np.sort(values)
            ax.step(ordered, np.arange(1, ordered.size + 1) / ordered.size * 100, where="post", color=style.INK)
            ax.axvline(mark, color=style.INK_SECONDARY, lw=0.8, ls=":")
            ax.set_xlabel(label)
            ax.set_ylabel("Spectra (%)")
            ax.set_ylim(0, 102)
            ax.set_yticks([0, 50, 100])
        ax_d.set_xlim(min(-0.05, float(data["exponent"].min())), max(1.7, float(data["exponent"].max())))
        ax_e.set_xlim(min(0.7, float(data["r2"].min())), 1.0)

        for ax, letter in ((ax_a, "A"), (ax_b, "B")):
            style.panel_label(ax, letter)
        # C, D and E have tick labels or a two-line axis label where the letter would go.
        style.panel_label(ax_c, "C", offset_points=(-58.0, 14.0))
        for ax, letter in ((ax_d, "D"), (ax_e, "E")):
            style.panel_label(ax, letter, offset_points=(-48.0, 16.0))
    return fig


def render_band_choice(
    project: ProjectConfig,
    *,
    streams: Sequence[str],
    settings: BandChoiceConfig,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """For each stream, specparam's group report of the model and the band-choice figure, under ``fig``."""
    from specparam import SpectralGroupModel

    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    inputs, outputs = [], []
    for stream in streams:
        entities = dict(stream=stream, lock=settings.lock, aperiodic=settings.aperiodic)
        bands = require_file(layout.band_choice(**entities, suffix="bands", extension=".h5"),
                             purpose=f"{stream} band choice; run `analyze band-choice` first")
        model_path = require_file(layout.band_choice(**entities, suffix="specparam", extension=".json"),
                                  purpose=f"{stream} specparam model; run `analyze band-choice` first")
        report = destination.band_choice(**entities, suffix="specparam", extension=".pdf", datatype="fig")
        report.parent.mkdir(parents=True, exist_ok=True)
        model = SpectralGroupModel()
        model.load(model_path.name, str(model_path.parent))
        model.save_report(report.name, str(report.parent))

        figure = plot_band_choice(bands)
        written = []
        for extension in FORMATS:
            target = destination.band_choice(**entities, suffix="bands", extension=extension, datatype="fig")
            figure.savefig(target, dpi=style.SAVEFIG_DPI)
            written.append(target)
        save_sidecar(written[0], {
            "stage": "band_choice_report", "stream": stream, **settings.__dict__, "formats": list(FORMATS),
            "inputs": [str(bands), str(model_path)], "specparam_report": str(report),
        })
        inputs += [bands, model_path]
        outputs += [report, *written]
    return WorkflowResult(stage="band_choice_report", inputs=tuple(inputs), outputs=tuple(outputs),
                          settings={"streams": list(streams), **settings.__dict__})


__all__ = ["plot_band_choice", "render_band_choice"]
