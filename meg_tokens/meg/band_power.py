"""Band power in source space, from the continuous cleaned recordings, through
MNE-Python. Specification and rationale: ``docs/meg.md`` (Decisions, Band power).

    Raw.filter -> Raw.apply_hilbert                        (sensors, per band)
    -> Epochs at the kept epochs' events
    -> apply_inverse_epochs, sLORETA, three orientations    (complex, per trial)
    -> cortex: normal component, |z|^2; deep: sum of |z|^2 over x, y, z
    -> extract_label_time_course (mean of power)            (regions, structures)
    -> SourceMorph.apply                                    (fsaverage, power)

The inverse is linear, so the analytic signal filtered and transformed on the
sensors is the analytic signal of every source. Filtering the whole run gives
every trial the full span of the filter; samples within half a filter length
of the recording's ends or of a flux jump are invalid. MNE refuses a normal
orientation on a mixed source space, so the estimate is taken with all three
orientations and the cortex projected onto its normals.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections import deque
from itertools import repeat
from pathlib import Path

import mne
import numpy as np

from meg_tokens.meg.epoching import WINDOWS
from meg_tokens.meg.sources import LAMBDA2, METHOD, VOLUME_LABELS

BANDS = {
    "theta": (5.0, 8.0),
    "alpha": (9.0, 13.0),
    "beta": (16.0, 24.0),
    "gamma": (30.0, 60.0),
    "highgamma": (60.0, 90.0),
}
# Transition widths below and above each band, in Hz. A zero-phase FIR is down
# 6 dB half a transition outside the band, so neighbouring bands meet without
# overlapping: theta 4.5-8.5, alpha 8.5-14, beta 15-26, gamma 28-61, high gamma
# 59-92.5 Hz. Only gamma and high gamma, which share the 60 Hz edge, meet over
# a band (59-61 Hz). MNE's default transitions make them share 52-68 Hz.
TRANSITIONS = {
    "theta": (1.0, 1.0),
    "alpha": (1.0, 2.0),
    "beta": (2.0, 4.0),
    "gamma": (4.0, 2.0),
    "highgamma": (2.0, 5.0),
}
LOCKS = ("go", "decision", "enter")
PARCELLATION = "HCPMMP1"
# Steps the filter spreads into the samples around them; the trunk repairs
# flux jumps, but what is left of one is still a step.
INVALID_NEAR = ("BAD_jump",)


def _filter_kwargs(band: str) -> dict:
    low, high = BANDS[band]
    l_trans, h_trans = TRANSITIONS[band]
    return dict(
        l_freq=low, h_freq=high, l_trans_bandwidth=l_trans, h_trans_bandwidth=h_trans,
        filter_length="auto", phase="zero", fir_window="hamming", fir_design="firwin",
    )


def half_length(sfreq: float, band: str) -> int:
    """Samples on each side of a sample that its filtered value depends on."""
    return mne.filter.create_filter(None, sfreq, **_filter_kwargs(band), verbose=False).size // 2


def analytic(raw: mne.io.BaseRaw, band: str, *, n_jobs: int = 1) -> mne.io.BaseRaw:
    """The magnetometers' band-passed analytic signal, as a complex Raw."""
    return (
        raw.copy().pick("mag")
        .filter(**_filter_kwargs(band), pad="reflect_limited", n_jobs=n_jobs, verbose=False)
        .apply_hilbert(envelope=False, n_jobs=n_jobs, verbose=False)
    )


def invalid_samples(raw: mne.io.BaseRaw, half: int) -> np.ndarray:
    """Samples of the recording within ``half`` samples of its ends or of a flux jump."""
    invalid = np.zeros(raw.n_times, dtype=bool)
    invalid[:half] = True
    invalid[raw.n_times - half:] = True
    annotations = raw.annotations
    near = np.isin(annotations.description, INVALID_NEAR)
    if near.any():
        onsets = raw.time_as_index(annotations.onset[near], use_rounding=True, origin=annotations.orig_time)
        ends = raw.time_as_index(annotations.onset[near] + annotations.duration[near], use_rounding=True,
                                 origin=annotations.orig_time)
        for onset, end in zip(onsets, ends):
            invalid[max(onset - half, 0): min(end + half + 1, raw.n_times)] = True
    return invalid


def window(sfreq: float, lock: str) -> np.ndarray:
    """The analysis window of a lock as sample offsets from the lock event, on MNE's grid.

    ``mne.Epochs`` cuts the same samples, so ``offsets / sfreq`` are the epochs' times.
    """
    tmin, tmax = WINDOWS[lock]
    return np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq)) + 1)


def morph_map(subject: str, subjects_dir: str | Path) -> Path:
    """The subject's morph map with ``fsaverage``, which must already exist.

    MNE writes a missing map into ``subjects_dir`` on first use, and many jobs
    starting at once for one subject would write and read it at the same time.
    :func:`prepare_morph_map` builds it beforehand, once.
    """
    folder = Path(subjects_dir) / "morph-maps"
    for name in (f"{subject}-fsaverage-morph.fif", f"fsaverage-{subject}-morph.fif"):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(
        f"No morph map between {subject} and fsaverage in {folder}; run `meg band-power-prepare` first"
    )


def prepare_morph_map(subject: str, subjects_dir: str | Path) -> Path:
    """Build the subject's morph map with ``fsaverage`` if it is missing (MNE writes it)."""
    mne.read_morph_map(subject, "fsaverage", subjects_dir=subjects_dir, verbose=False)
    return morph_map(subject, subjects_dir)


def hcp_labels(subject: str, subjects_dir: str | Path) -> list[mne.Label]:
    """The 360 HCP-MMP1 regions, from ``fsaverage`` onto the subject's cortex.

    The parcellation has a ``???`` label per hemisphere for the medial wall,
    which is not a region and is left out. Morphing writes the morph maps into
    ``subjects_dir`` the first time.
    """
    labels = [
        label for label in mne.read_labels_from_annot("fsaverage", PARCELLATION, subjects_dir=subjects_dir, verbose=False)
        if not label.name.startswith("???")
    ]
    if subject == "fsaverage":
        return labels
    return mne.morph_labels(labels, subject, "fsaverage", subjects_dir, verbose=False)


def deep_labels(subject: str, subjects_dir: str | Path) -> tuple[str, list[str]]:
    """The deep structures as MNE's volume labels: the subject's ``aseg`` and the structures' names."""
    return str(Path(subjects_dir) / subject / "mri" / "aseg.mgz"), list(VOLUME_LABELS)


def fsaverage_morph(src: mne.SourceSpaces, subjects_dir: str | Path) -> mne.SourceMorph:
    """The morph of the subject's cortex onto ``fsaverage`` ``oct6``.

    MNE cannot morph the mixed source space (its volume has no interpolator);
    the cortex alone morphs, and is applied here to power, never to signed
    time courses.
    """
    target = mne.setup_source_space("fsaverage", "oct6", subjects_dir=subjects_dir, add_dist=False, verbose=False)
    return mne.compute_source_morph(
        mne.SourceSpaces(list(src[:2])), src[0]["subject_his_id"], "fsaverage", subjects_dir=subjects_dir,
        src_to=target, verbose=False,
    )


def _power(stc):
    """|z|^2 of a complex estimate; a vector estimate's three orientations summed."""
    power = np.abs(stc.data) ** 2
    if power.ndim == 3:
        return mne.VolSourceEstimate(power.sum(axis=1), stc.vertices, stc.tmin, stc.tstep, stc.subject)
    return mne.SourceEstimate(power, stc.vertices, stc.tmin, stc.tstep, stc.subject)


def trial_power(
    epochs: mne.BaseEpochs,
    inv: mne.minimum_norm.InverseOperator,
    labels: list[mne.Label],
    deep: tuple[str, list[str]],
    morph: mne.SourceMorph | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """For each trial of complex epochs, its region and deep-structure power and its fsaverage power.

    ``inv`` is prepared (``mne.minimum_norm.prepare_inverse_operator``). A
    region's power is the mean of its sources' |z|^2 (``mode="mean"`` on the
    power estimate, so nothing cancels); a deep structure's, the mean over its
    sources of the three orientations' summed |z|^2, which does not depend on
    the axes. Yields (regions + structures) x times, and fsaverage sources x
    times when ``morph`` is given. A region with no source is NaN.
    """
    src = inv["src"]
    estimates = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, LAMBDA2, METHOD, pick_ori="vector", prepared=True, return_generator=True, verbose=False,
    )
    # Each estimate (about 1 GB: three orientations, complex) is read once and
    # let go; what the other streams need from it waits in a one-item queue.
    # itertools.tee would keep up to 57 estimates alive, in its blocks.
    deep_waiting, cortex_waiting = deque(), deque()

    def cortex() -> Iterator[mne.SourceEstimate]:
        for stc in estimates:
            deep_waiting.append(_power(stc.volume()))
            power = _power(stc.surface().project("normal", src[:2])[0])
            if morph is not None:
                cortex_waiting.append(power)
            yield power

    deep_power = (deep_waiting.popleft() for _ in repeat(None))
    fsaverage = repeat(None) if morph is None else (morph.apply(cortex_waiting.popleft()).data for _ in repeat(None))
    regions = mne.extract_label_time_course(
        cortex(), labels, src[:2], mode="mean", allow_empty=True, return_generator=True, verbose=False,
    )
    structures = mne.extract_label_time_course(
        deep_power, deep, src[2:], mode="mean", mri_resolution=False, allow_empty=True,
        return_generator=True, verbose=False,
    )
    empty = np.r_[[len(label.restrict(src[:2]).vertices) == 0 for label in labels], np.zeros(len(deep[1]), bool)]
    for region, structure, average in zip(regions, structures, fsaverage):
        power = np.concatenate([region, structure])
        power[empty] = np.nan
        yield power, average



def deep_trial_power(epochs: mne.BaseEpochs, inv: mne.minimum_norm.InverseOperator) -> Iterator[np.ndarray]:
    """For each trial of complex epochs, the power of every deep source, sources x times.

    A deep source has no surface to be normal to: its power is the sum over x,
    y and z of |z|^2, which does not depend on the axes. ``inv`` is prepared;
    MNE applies the whole operator, and only the volume's estimate is kept.
    """
    for stc in mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, LAMBDA2, METHOD, pick_ori="vector", prepared=True, return_generator=True, verbose=False,
    ):
        yield _power(stc.volume()).data


def fsaverage_vertices(subjects_dir: str | Path) -> list[np.ndarray]:
    """The ``fsaverage`` ``oct6`` vertices the fsaverage power is on, left then right."""
    target = mne.setup_source_space("fsaverage", "oct6", subjects_dir=subjects_dir, add_dist=False, verbose=False)
    return [hemisphere["vertno"] for hemisphere in target]

__all__ = [
    "BANDS", "TRANSITIONS", "LOCKS", "PARCELLATION", "INVALID_NEAR",
    "half_length", "analytic", "invalid_samples", "window",
    "morph_map", "prepare_morph_map", "hcp_labels", "deep_labels", "fsaverage_morph", "trial_power",
    "deep_trial_power", "fsaverage_vertices",
]
