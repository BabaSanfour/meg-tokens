"""Myographic (EMG) artifact screening across subjects.

Reproduces the EMG columns in docs/meg.md
("MEG-signal evidence"): the broadband power ratio z-scores and the ICA
muscle-component ranking that together identify H04 -- which is *not* one of
the four excluded subjects -- as the strongest EMG outlier in the cohort,
while H06/H07/H10 show no EMG signal.

Two independent measures per subject, on Slow/Fast runs only:

  1. broadband power ratio, mean(40-100 Hz) / mean(1-20 Hz) over magnetometers
  2. count of ICA components that look muscular, scored as
     (fraction of the spatial pattern on frontal/temporal sensors)
     x (how flat the component spectrum is vs the neural ~1/f falloff)

Both read a bounded window per run rather than whole recordings, which is
what the doc's "60-120 s windows" limitation refers to.

    python scripts/qc/meg_emg_qc.py <raw-root>
"""

import argparse
import re
from pathlib import Path

import numpy as np

SLOWFAST_TRIAL_DURATION = 315.0
# CTF sensor prefixes over frontal and temporal cortex, where neck/jaw/ocular
# muscle activity projects most strongly.
EDGE_PREFIXES = ("MLF", "MRF", "MZF", "MLT", "MRT")
MUSCLE_INDEX_THRESHOLD = 0.3
WINDOW_SECONDS = 60.0


def slowfast_runs(data_root, subject, limit=None):
    """Slow/Fast .ds directories for one subject, oldest first."""
    runs = []
    for ds_dir in sorted(data_root.glob(f"{subject}_*.ds")):
        hist_files = list(ds_dir.glob("*.hist"))
        if not hist_files:
            continue
        match = re.search(
            r"Trial duration:\s*(\S+)", hist_files[0].read_text(errors="replace")
        )
        if match and float(match.group(1)) == SLOWFAST_TRIAL_DURATION:
            runs.append(ds_dir)
        if limit and len(runs) >= limit:
            break
    return runs


def load_centre_window(ds_dir, seconds=WINDOW_SECONDS):
    """Magnetometer data from a window at the centre of one run."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_ctf(str(ds_dir), preload=False, verbose=False)
    raw.pick("mag")
    sfreq = raw.info["sfreq"]
    midpoint = raw.n_times // 2
    half = int(seconds / 2 * sfreq)
    raw.crop(
        tmin=max(0, midpoint - half) / sfreq,
        tmax=min(raw.n_times - 1, midpoint + half) / sfreq,
    )
    raw.load_data(verbose=False)
    return raw


def broadband_ratio(raw):
    """High-frequency power relative to low, averaged over channels."""
    sfreq = raw.info["sfreq"]
    spectrum = raw.compute_psd(
        fmin=1, fmax=140, method="welch", n_fft=int(sfreq * 2), verbose=False
    )
    freqs, power = spectrum.freqs, spectrum.get_data()
    low = power[:, (freqs >= 1) & (freqs < 20)].mean()
    high = power[:, (freqs >= 40) & (freqs < 100)].mean()
    return float(high / low)


def count_muscle_components(raw, n_components=20):
    """Number of ICA components with a muscle-like spectrum and topography."""
    import mne
    from mne.preprocessing import ICA

    prepared = raw.copy().resample(250, verbose=False).filter(1.0, 100.0, verbose=False)
    ica = ICA(n_components=n_components, method="fastica", max_iter=200, random_state=97)
    ica.fit(prepared, verbose=False)

    sources = ica.get_sources(prepared).get_data()
    patterns = ica.get_components()
    edge = [i for i, name in enumerate(prepared.ch_names) if name[:3] in EDGE_PREFIXES]

    count, strongest = 0, 0.0
    for component in range(sources.shape[0]):
        power, freqs = mne.time_frequency.psd_array_welch(
            sources[component][None, :], sfreq=250.0, fmin=5, fmax=100,
            n_fft=500, verbose=False,
        )
        usable = power[0] > 0
        # Neural spectra fall off steeply (~1/f, slope near -2); muscle activity
        # is broadband, so its slope sits near zero or above.
        slope = np.polyfit(np.log10(freqs[usable]), np.log10(power[0][usable]), 1)[0]
        weights = np.abs(patterns[:, component])
        edge_fraction = weights[edge].sum() / weights.sum()
        muscle_index = edge_fraction * max(0.0, slope + 2.0)
        strongest = max(strongest, muscle_index)
        if muscle_index > MUSCLE_INDEX_THRESHOLD:
            count += 1
    return count, strongest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--subjects", nargs="*", help="Default: H01-H32")
    parser.add_argument(
        "--runs-per-subject", type=int, default=2,
        help="Slow/Fast runs to pool per subject (default 2)",
    )
    args = parser.parse_args()

    subjects = args.subjects or [f"H{i:02d}" for i in range(1, 33)]

    rows = []
    for subject in subjects:
        runs = slowfast_runs(args.data_root, subject, limit=args.runs_per_subject)
        if not runs:
            print(f"  {subject}: no Slow/Fast runs found, skipping")
            continue
        ratios = [broadband_ratio(load_centre_window(run)) for run in runs]
        # dev_head_t differs between runs; ignore it, we never localise sources here.
        import mne

        pooled = (
            mne.concatenate_raws(
                [load_centre_window(run, WINDOW_SECONDS / 2) for run in runs],
                on_mismatch="ignore", verbose=False,
            )
            if len(runs) > 1
            else load_centre_window(runs[0])
        )
        n_muscle, strongest = count_muscle_components(pooled)
        rows.append((subject, float(np.mean(ratios)), n_muscle, strongest))
        print(f"  {subject} done", flush=True)

    ratios = np.array([r[1] for r in rows])
    mean, sd = ratios.mean(), ratios.std(ddof=1)
    print(f"\nBroadband ratio population: mean={mean:.4f} sd={sd:.4f}\n")
    print(f"  {'subject':<9}{'ratio':>9}{'z':>8}{'muscle ICs':>12}{'top index':>11}")
    for subject, ratio, n_muscle, strongest in sorted(rows, key=lambda r: -r[1]):
        print(
            f"  {subject:<9}{ratio:>9.4f}{(ratio - mean) / sd:>8.2f}"
            f"{n_muscle:>12d}{strongest:>11.3f}"
        )


if __name__ == "__main__":
    main()
