"""Typed settings for preprocessing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# The events epochs are locked to; their windows are in meg_tokens.meg.epoching.
EPOCH_LOCKS = ("go", "enter", "feedback", "decision")
STREAMS = ("dss", "ica")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing settings. ``streams`` names the cleaners to run per recording.

    The band and every cleaner parameter are fixed by the specification in
    ``meg/preprocessing.py`` rather than settable here: the streams are
    only comparable across recordings if no run was cleaned differently.
    """

    streams: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.streams is not None and not self.streams:
            raise ValueError("streams must name at least one cleaner")


@dataclass(frozen=True)
class RawStagingConfig:
    """Settings for Stage 0 raw-BIDSification (media -> BIDS/sub-*/meg,beh)."""

    slowfast_nominal_duration: float = 315.0
    rt_nominal_duration: float = 135.0
    # Real nominal durations across all 412 subject sessions on the
    # dataset take exactly four values -- 135.0, 315.0, 360.0, 420.0
    # -- with a minimum 45s gap between clusters, so 20s leaves wide margin
    # without risking a false match (verified across this dataset).
    duration_tolerance: float = 20.0
    # Small slack for real-world Start-pulse-count logging jitter, kept
    # deliberately tight: documented per-run exceptions
    # (KNOWN_TRAILING_TRIAL_MISMATCHES) are whitelisted individually rather
    # than absorbed by loosening this general tolerance. Unlike
    # duration_tolerance, this value has not been empirically validated
    # against the real per-session pulse-vs-trial-count distribution.
    count_tolerance: int = 3
    # Inter-trial-interval fingerprint acceptance guards (see
    # meg_tokens.meg.raw_staging.fingerprint_candidates). On the real
    # dataset a correct match scores 0.39-0.57ms mean absolute error and
    # beats the runner-up by 144-892x, so these thresholds sit in the wide
    # empty gap between "unambiguously right" and anything else -- a
    # candidate that fails them is reported ambiguous for human review
    # rather than accepted as a best-available guess.
    fingerprint_max_error_ms: float = 5.0
    fingerprint_min_separation: float = 20.0

    def __post_init__(self) -> None:
        if self.slowfast_nominal_duration <= 0 or self.rt_nominal_duration <= 0:
            raise ValueError("Nominal durations must be positive")
        if self.duration_tolerance < 0:
            raise ValueError("duration_tolerance must be non-negative")
        if self.count_tolerance < 0:
            raise ValueError("count_tolerance must be non-negative")
        if self.fingerprint_max_error_ms <= 0:
            raise ValueError("fingerprint_max_error_ms must be positive")
        if self.fingerprint_min_separation < 1:
            raise ValueError("fingerprint_min_separation must be at least 1")


@dataclass(frozen=True)
class EpochingConfig:
    """Which locks to epoch, from which preprocessing streams."""

    locks: Tuple[str, ...] = EPOCH_LOCKS
    streams: Tuple[str, ...] = STREAMS

    def __post_init__(self) -> None:
        for name, values, allowed in (("lock", self.locks, EPOCH_LOCKS), ("stream", self.streams, STREAMS)):
            unknown = sorted(set(values) - set(allowed))
            if unknown or not values:
                raise ValueError(f"{name}s must be a non-empty subset of {list(allowed)}, got {list(values)}")


@dataclass(frozen=True)
class CoregistrationConfig:
    """How the MEG-MRI fit is run, and what a subject must reach to pass QC.

    ``template_subject`` is the anatomy borrowed by subjects with no FreeSurfer
    reconstruction of their own. ``scale_template`` fits that template to the
    subject's head shape instead of using it at its own size, which matters:
    unscaled, those subjects keep barely half their digitized points within
    reach of the scalp.
    """

    coarse_iterations: int = 6
    fine_iterations: int = 20
    omit_distance: float = 0.005
    template_subject: str = "fsaverage"
    scale_template: bool = True
    # Flagged, never dropped: a subject past either bound is reported for
    # inspection, because a bad coregistration is a thing to look at, not a
    # thing to silently exclude.
    max_median_distance_mm: float = 4.0
    max_landmark_residual_mm: float = 8.0

    def __post_init__(self) -> None:
        if self.omit_distance <= 0:
            raise ValueError("omit_distance must be positive")
        if self.coarse_iterations < 0 or self.fine_iterations < 0:
            raise ValueError("ICP iteration counts must not be negative")
        if not self.template_subject.strip():
            raise ValueError("template_subject must not be empty")


@dataclass(frozen=True)
class PowerConfig:
    run: str
    condition: Optional[str] = None
    alignment: str = "go"
    source_method: str = "dSPM"
    method: str = "hilbert"
    bands: Optional[Tuple[Tuple[str, float, float], ...]] = None
    width: int = 400
    step: int = 110
    n_jobs: int = 1
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None
    baseline_method: str = "percent"

    def __post_init__(self) -> None:
        if self.alignment not in EPOCH_LOCKS:
            raise ValueError(f"Unknown alignment event: {self.alignment}")
        if self.method not in {"hilbert", "morlet", "multitaper"}:
            raise ValueError(f"Unknown power method: {self.method}")
        if self.width <= 0 or self.step <= 0:
            raise ValueError("width and step must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be zero")
        if self.bands:
            for name, fmin, fmax in self.bands:
                if not name or fmin <= 0 or fmax <= fmin:
                    raise ValueError(f"Invalid frequency band: {(name, fmin, fmax)}")


@dataclass(frozen=True)
class SpectralConfig:
    """Each subject's sensor spectra: the magnetometers around one lock (meg.md §2, Bands).

    Welch's method over the lock's whole analysis window, trial-averaged over
    every Fast and Slow run; the defaults are the preprint's band-choice input
    (press lock, -1.5 to +2.0 s, 2 s segments at 600 Hz: 0.5 Hz resolution).
    """

    lock: str = "enter"
    fmin: float = 1.0
    fmax: float = 90.0
    n_fft: int = 1200
    n_overlap: int = 600

    def __post_init__(self) -> None:
        if self.fmin < 0 or self.fmax <= self.fmin:
            raise ValueError("Spectral bounds must satisfy 0 <= fmin < fmax")
        if self.n_fft <= 0 or not 0 <= self.n_overlap < self.n_fft:
            raise ValueError("n_fft must be positive and 0 <= n_overlap < n_fft")


@dataclass(frozen=True)
class BandChoiceConfig:
    """The preprint's band choice from spectral peaks (its Fig S1), and the checks of it.

    One specparam model of every subject's sensor spectra with FOOOF's
    defaults over ``fmin``-``fmax``, the aperiodic component ``fixed`` (the
    preprint's) or with a ``knee``; peak centre frequencies counted in bins of
    ``bin_hz``, each subject weighted equally; the bands where the histogram is
    above its mean bin; ``n_resamples`` resamplings of subjects for the edges'
    intervals; spectra fitted below ``min_r2`` counted.
    """

    lock: str = "enter"
    fmin: float = 1.0
    fmax: float = 35.0
    aperiodic: str = "fixed"
    bin_hz: float = 1.0
    n_resamples: int = 2000
    min_r2: float = 0.9

    def __post_init__(self) -> None:
        if self.fmin <= 0 or self.fmax <= self.fmin:
            raise ValueError("Band-choice bounds must satisfy 0 < fmin < fmax")
        if self.aperiodic not in {"fixed", "knee"}:
            raise ValueError(f"aperiodic must be 'fixed' or 'knee', got {self.aperiodic!r}")
        if self.bin_hz <= 0 or self.n_resamples < 1 or not 0 < self.min_r2 < 1:
            raise ValueError("bin_hz must be positive, n_resamples at least 1 and min_r2 in (0, 1)")


@dataclass(frozen=True)
class HilbertConfig:
    conditions: Tuple[str, ...]
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    labels: Optional[Tuple[str, ...]] = None
    bands: Optional[Tuple[Tuple[str, float, float], ...]] = None
    features: Tuple[str, ...] = ("amplitude", "power", "phase", "sigfilt")
    sfreq: Optional[float] = None
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("At least one condition is required")
        if self.alignment not in EPOCH_LOCKS:
            raise ValueError(f"Unknown alignment event: {self.alignment}")
        valid = {"amplitude", "power", "phase", "sigfilt", "filtered"}
        invalid = sorted(set(self.features) - valid)
        if invalid:
            raise ValueError(f"Unknown Hilbert features: {invalid}")


@dataclass(frozen=True)
class PACConfig:
    conditions: Tuple[str, ...]
    phase_bands: Tuple[str, ...]
    amplitude_bands: Tuple[str, ...]
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    method: str = "modulation_index"
    n_bins: int = 18
    time_window: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not self.conditions or not self.phase_bands or not self.amplitude_bands:
            raise ValueError("conditions, phase_bands, and amplitude_bands are required")
        if self.method != "modulation_index":
            raise ValueError(f"Unknown PAC method: {self.method}")
        if self.n_bins < 3:
            raise ValueError("n_bins must be at least 3")
        if self.time_window and self.time_window[1] <= self.time_window[0]:
            raise ValueError("PAC time window stop must be greater than start")


@dataclass(frozen=True)
class ConnectivityConfig:
    conditions: Tuple[str, ...]
    alignment: str = "enter"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    labels: Optional[Tuple[str, ...]] = None
    bands: Tuple[Tuple[str, float, float], ...] = (
        ("delta", 2.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 15.0),
        ("beta", 15.0, 30.0),
    )
    method: str = "imcoh"
    mode: str = "fourier"
    sfreq: Optional[float] = None
    before_window: Tuple[float, float] = (0.7, 1.4)
    after_window: Tuple[float, float] = (1.6, 2.3)
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("At least one condition is required")
        for name, fmin, fmax in self.bands:
            if not name or fmin <= 0 or fmax <= fmin:
                raise ValueError(f"Invalid connectivity band: {(name, fmin, fmax)}")
        for window in (self.before_window, self.after_window):
            if window[1] <= window[0]:
                raise ValueError("Connectivity window stop must be greater than start")


@dataclass(frozen=True)
class StatisticsConfig:
    conditions: Tuple[str, str] = ("Fast", "Slow")
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    runs_condition_1: Optional[Tuple[str, ...]] = None
    runs_condition_2: Optional[Tuple[str, ...]] = None
    permutations: int = 1000
    tail: int = 0
    alpha: float = 0.05
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if len(self.conditions) != 2 or self.conditions[0] == self.conditions[1]:
            raise ValueError("Statistics require two distinct conditions")
        if self.permutations <= 0:
            raise ValueError("permutations must be positive")
        if self.tail not in {-1, 0, 1}:
            raise ValueError("tail must be -1, 0, or 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")


@dataclass(frozen=True)
class LateralizedStatisticsConfig:
    condition: str = "Easy"
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    runs: Optional[Tuple[str, ...]] = None
    permutations: int = 1000
    tail: int = 0
    alpha: float = 0.05
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not self.condition:
            raise ValueError("condition is required")
        if self.permutations <= 0:
            raise ValueError("permutations must be positive")
        if self.tail not in {-1, 0, 1}:
            raise ValueError("tail must be -1, 0, or 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")


@dataclass(frozen=True)
class DecodingConfig:
    conditions: Tuple[str, ...] = ("Fast", "Slow")
    input_conditions: Optional[Tuple[str, ...]] = None
    feature_source: str = "erp"
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: Optional[str] = "HCPMMP1"
    band: Optional[str] = None
    power_method: str = "hilbert"
    runs: Optional[Tuple[str, ...]] = None
    labels: Optional[Tuple[str, ...]] = None
    lateralize: bool = False
    class_column: Optional[str] = None
    class_values: Optional[Tuple[str, ...]] = None
    permutations: int = 0
    n_jobs: int = 4

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("Decoding requires at least two conditions")
        if self.feature_source not in {"erp", "power"}:
            raise ValueError("feature_source must be 'erp' or 'power'")
        if self.feature_source == "power" and self.band is None:
            raise ValueError("band is required for power decoding")
        if self.class_column and not self.class_values:
            raise ValueError("class_values are required with class_column")
        if self.permutations < 0:
            raise ValueError("permutations must be non-negative")


@dataclass(frozen=True)
class SpatialDecodingConfig:
    conditions: Tuple[str, ...] = ("Fast", "Slow")
    permutations: int = 0
    n_jobs: int = 4

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("Spatial decoding requires at least two conditions")
        if self.permutations < 0:
            raise ValueError("permutations must be non-negative")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


# What a trajectory PCA reads: band power as each of its means' three forms, or
# the signed event-related fields; and where (fields also on fsaverage).
TRAJECTORY_MEASURES = ("power", "log10", "db", "erp")
TRAJECTORY_SPACES = ("HCPMMP1", "fsaverage", "fsaverageaseg")


@dataclass(frozen=True)
class TrajectoryConfig:
    """One PCA trajectory analysis (meg.md §2, Analyses).

    ``fit`` is ``grandaverage`` (one PCA of the grand average of the subjects'
    condition means, each subject projected uncentred, as the preprint's code
    did) or ``pooled`` (one PCA of every subject's condition means pooled at
    equal weight, projected centred, as the trajectory primer does).
    ``measure`` is band power as ``power`` (linear), ``log10`` (centred per
    subject, band and region) or ``db`` (against each trial's pre-token
    baseline), or ``erp``, the signed fields, in the HCP-MMP1 regions or on
    ``fsaverage``. ``window`` (seconds from the lock) restricts the fit and the
    trajectories to part of the lock's analysis window; ``None`` keeps all of it.
    ``deep`` adds the 17 deep structures' band power to the 360 HCP-MMP1
    regions, so their loadings can be read beside the cortex's.
    ``fsaverageaseg`` is the deep sources' band power on fsaverage's grid, one
    PCA per ``structure``: a structure's name (``Left-Caudate``) or, without
    its side, both sides together (``Caudate``). ``induced`` reads the second
    copy of band power, with the task events' responses taken out.
    """

    fit: str = "pooled"
    measure: str = "log10"
    space: str = "HCPMMP1"
    n_components: int = 20
    window: Optional[Tuple[float, float]] = None
    deep: bool = False
    structure: Optional[str] = None
    induced: bool = False

    def __post_init__(self) -> None:
        if self.fit not in ("grandaverage", "pooled"):
            raise ValueError(f"fit must be 'grandaverage' or 'pooled', got {self.fit!r}")
        if self.measure not in TRAJECTORY_MEASURES:
            raise ValueError(f"measure must be one of {list(TRAJECTORY_MEASURES)}, got {self.measure!r}")
        if self.space not in TRAJECTORY_SPACES:
            raise ValueError(f"space must be one of {list(TRAJECTORY_SPACES)}, got {self.space!r}")
        if (self.space == "fsaverageaseg") != (self.structure is not None):
            raise ValueError("the deep sources' grid (fsaverageaseg) takes one structure, and only it does")
        if self.space == "fsaverageaseg" and self.measure == "erp":
            raise ValueError("the deep sources' grid holds band power only")
        if self.induced and (self.measure == "erp" or self.space == "fsaverageaseg"):
            raise ValueError("the induced copy is of band power, in the HCP-MMP1 regions and on fsaverage")
        if self.deep and (self.space != "HCPMMP1" or self.measure == "erp"):
            raise ValueError("deep structures enter the band power of the HCP-MMP1 regions only")
        if self.window is not None and not self.window[0] < self.window[1]:
            raise ValueError(f"window must run forward in time, got {self.window}")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
