"""Typed settings for preprocessing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


SOURCE_STAGES = ("cov", "bem", "src", "fwd", "inv", "apply")


@dataclass(frozen=True)
class PreprocessingConfig:
    high_pass: float = 0.5
    low_pass: float = 150.0
    notch_freqs: Optional[Tuple[float, ...]] = None
    run_ica: bool = False
    ica_exclude: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.high_pass < 0:
            raise ValueError("high_pass must be non-negative")
        if self.low_pass <= self.high_pass:
            raise ValueError("low_pass must be greater than high_pass")
        if self.notch_freqs is not None and any(value <= 0 for value in self.notch_freqs):
            raise ValueError("notch frequencies must be positive")


@dataclass(frozen=True)
class RawStagingConfig:
    """Settings for Stage 0 raw-BIDSification (media -> BIDS/sub-*/meg,beh)."""

    slowfast_nominal_duration: float = 315.0
    rt_nominal_duration: float = 135.0
    duration_tolerance: float = 20.0
    count_tolerance: int = 3

    def __post_init__(self) -> None:
        if self.slowfast_nominal_duration <= 0 or self.rt_nominal_duration <= 0:
            raise ValueError("Nominal durations must be positive")
        if self.duration_tolerance < 0:
            raise ValueError("duration_tolerance must be non-negative")
        if self.count_tolerance < 0:
            raise ValueError("count_tolerance must be non-negative")


@dataclass(frozen=True)
class EpochingConfig:
    tmin: float = -0.5
    tmax: float = 2.0
    alignment: str = "go"

    def __post_init__(self) -> None:
        alignment = self.alignment.lower()
        if alignment not in {"go", "enter", "feedback"}:
            raise ValueError(f"Unknown alignment event: {self.alignment}")
        if self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin")
        object.__setattr__(self, "alignment", alignment)


@dataclass(frozen=True)
class SourceConfig:
    stages: Tuple[str, ...] = SOURCE_STAGES
    spacing: str = "oct6"
    volume_labels: Optional[Tuple[str, ...]] = None
    volume_pos: float = 5.0
    run: Optional[str] = None
    condition: Optional[str] = None
    alignment: str = "go"
    method: str = "dSPM"
    snr: float = 1.0

    def __post_init__(self) -> None:
        invalid = sorted(set(self.stages) - set(SOURCE_STAGES))
        if invalid:
            raise ValueError(f"Unknown source stages: {invalid}")
        if not self.stages:
            raise ValueError("At least one source stage is required")
        if any(stage in self.stages for stage in ("fwd", "inv", "apply")) and self.run is None:
            raise ValueError("run is required for fwd, inv, and apply source stages")
        alignment = self.alignment.lower()
        if alignment not in {"go", "enter", "feedback"}:
            raise ValueError(f"Unknown alignment event: {self.alignment}")
        if self.volume_pos <= 0:
            raise ValueError("volume_pos must be positive")
        if self.snr <= 0:
            raise ValueError("snr must be positive")
        object.__setattr__(self, "alignment", alignment)


@dataclass(frozen=True)
class ERPConfig:
    run: str
    condition: Optional[str] = None
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: str = "HCPMMP1"
    feature_space: str = "parcellated"
    hemi: str = "both"
    label_subject: Optional[str] = None
    spacing: Optional[str] = None
    label_mode: str = "mean"
    max_duration_samples: int = 400
    cutoff_before_enter_ms: float = 300.0
    min_rt_ms: float = 100.0

    def __post_init__(self) -> None:
        if self.alignment not in {"go", "enter", "feedback"}:
            raise ValueError(f"Unknown alignment event: {self.alignment}")
        if self.feature_space not in {"parcellated", "all_source", "volume"}:
            raise ValueError(f"Unknown feature space: {self.feature_space}")
        if self.hemi not in {"left", "right", "both"}:
            raise ValueError(f"Unknown hemisphere selection: {self.hemi}")
        if self.max_duration_samples <= 0:
            raise ValueError("max_duration_samples must be positive")


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
        if self.alignment not in {"go", "enter", "feedback"}:
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
    run: Optional[str] = None
    condition: Optional[str] = None
    alignment: Optional[str] = None
    fmin: float = 1.0
    fmax: float = 100.0
    method: str = "welch"
    n_fft: int = 2048
    n_overlap: int = 150
    n_jobs: int = 1
    fit_model: bool = True

    def __post_init__(self) -> None:
        if self.fmin < 0 or self.fmax <= self.fmin:
            raise ValueError("Spectral bounds must satisfy 0 <= fmin < fmax")
        if self.method not in {"welch", "multitaper"}:
            raise ValueError(f"Unknown PSD method: {self.method}")
        if self.n_fft <= 0 or self.n_overlap < 0:
            raise ValueError("n_fft must be positive and n_overlap non-negative")


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
        if self.alignment not in {"go", "enter", "feedback"}:
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


@dataclass(frozen=True)
class DecompositionConfig:
    conditions: Tuple[str, ...] = ("Fast", "Slow")
    analysis: str = "pca"
    feature_source: str = "erp"
    alignment: str = "go"
    source_method: str = "dSPM"
    parc: Optional[str] = "HCPMMP1"
    band: Optional[str] = None
    power_method: str = "hilbert"
    labels: Optional[Tuple[str, ...]] = None
    lateralize: bool = False
    average_unit: str = "subject"
    transform: Optional[str] = None
    n_components: int = 20
    min_variance: Optional[float] = None
    fit_time_range: Optional[Tuple[float, float]] = None
    project_centered: bool = False
    marginalize_cols: Optional[Tuple[str, ...]] = None
    dpca_labels: Optional[str] = None

    def __post_init__(self) -> None:
        if self.analysis not in {"pca", "dpca"}:
            raise ValueError("analysis must be 'pca' or 'dpca'")
        if self.feature_source not in {"erp", "power"}:
            raise ValueError("feature_source must be 'erp' or 'power'")
        if self.feature_source == "power" and self.band is None:
            raise ValueError("band is required for power decomposition")
        if self.average_unit not in {"subject", "trial"}:
            raise ValueError("average_unit must be 'subject' or 'trial'")
        if self.transform not in {None, "sqrt", "signed-sqrt"}:
            raise ValueError(f"Unknown transform: {self.transform}")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.analysis == "dpca" and not self.marginalize_cols:
            raise ValueError("marginalize_cols are required for dPCA")
