import pytest
import numpy as np
import mne
import pandas as pd
import os
from meg_tokens.meg.sources import (
    compute_noise_covariance,
    setup_bem_solution,
    setup_mixed_source_space,
    compute_forward_solution,
    build_inverse_operator,
    apply_inverse_operator,
    save_source_estimates,
    save_source_space,
    source_derivative_path,
)


class Recorder:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


class RawStub:
    def __init__(self):
        self.info = {'ch_names': ['MEG001', 'MEG002']}
        self.picked = []

    def pick_channels(self, channels):
        self.picked.append(list(channels))


class SourceSpaceStub:
    def __init__(self):
        self.added = []

    def __iadd__(self, other):
        self.added.append(other)
        return self


def test_compute_noise_covariance(monkeypatch):
    raw = RawStub()
    cov = object()
    read_raw = Recorder(raw)
    compute_cov = Recorder(cov)
    monkeypatch.setattr("meg_tokens.meg.sources.os.path.exists", lambda path: True)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.io.read_raw_fif", read_raw)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.compute_raw_covariance", compute_cov)

    res = compute_noise_covariance('input_path.fif', ch_names=['MEG001'])

    assert read_raw.calls == [(('input_path.fif',), {'preload': True})]
    assert raw.picked == [['MEG001']]
    assert len(compute_cov.calls) == 1
    assert res is cov


def test_setup_bem_solution(monkeypatch):
    model = object()
    solution = object()
    make_model = Recorder(model)
    make_solution = Recorder(solution)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.make_bem_model", make_model)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.make_bem_solution", make_solution)

    res = setup_bem_solution('H1', 'subjects_dir')

    assert make_model.calls == [(
        (),
        {'subject': 'H1', 'ico': 4, 'conductivity': (0.3,), 'subjects_dir': 'subjects_dir'},
    )]
    assert make_solution.calls == [((model,), {})]
    assert res is solution


def test_setup_mixed_source_space(monkeypatch):
    surf_src = SourceSpaceStub()
    vol_src = SourceSpaceStub()
    setup_surf = Recorder(surf_src)
    setup_vol = Recorder(vol_src)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.setup_source_space", setup_surf)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.setup_volume_source_space", setup_vol)

    res = setup_mixed_source_space('H1', 'subjects_dir', volume_labels=None)
    assert setup_surf.calls == [(
        ('H1',),
        {'spacing': 'oct6', 'add_dist': False, 'subjects_dir': 'subjects_dir'},
    )]
    assert res is surf_src

    setup_surf.calls.clear()
    res_mixed = setup_mixed_source_space(
        'H1', 'subjects_dir', spacing='oct6', volume_labels=['Left-Putamen']
    )
    assert setup_vol.calls == [(
        ('H1',),
        {
            'mri': 'aseg.mgz',
            'pos': 5.0,
            'bem': None,
            'volume_label': ['Left-Putamen'],
            'subjects_dir': 'subjects_dir',
            'add_interpolator': True,
            'verbose': False,
        },
    )]
    assert surf_src.added == [vol_src]
    assert res_mixed is surf_src


def test_compute_forward_solution(monkeypatch):
    info = object()
    src = object()
    bem = object()
    fwd = object()
    make_fwd = Recorder(fwd)
    monkeypatch.setattr("meg_tokens.meg.sources.mne.make_forward_solution", make_fwd)

    res = compute_forward_solution(info, 'trans.fif', src, bem)

    assert make_fwd.calls == [((info,), {
        'trans': 'trans.fif',
        'src': src,
        'bem': bem,
        'mindist': 5.0,
        'meg': True,
        'eeg': False,
    })]
    assert res is fwd


def test_build_inverse_operator(monkeypatch):
    info = object()
    fwd = object()
    cov = object()
    inv = object()
    make_inv = Recorder(inv)
    monkeypatch.setattr("meg_tokens.meg.sources.make_inverse_operator", make_inv)

    res = build_inverse_operator(info, fwd, cov)

    assert make_inv.calls == [((info,), {
        'forward': fwd,
        'noise_cov': cov,
        'loose': dict(surface=0.2, volume=1.0),
        'depth': None,
        'verbose': True,
    })]
    assert res is inv


def test_apply_inverse_operator(monkeypatch):
    epochs = object()
    inv = object()
    stc = object()
    apply_inverse = Recorder(stc)
    monkeypatch.setattr("meg_tokens.meg.sources.apply_inverse_epochs", apply_inverse)

    res = apply_inverse_operator(epochs, inv, method='dSPM', snr=1.0)

    assert apply_inverse.calls == [((epochs,), {
        'inverse_operator': inv,
        'lambda2': 1.0,
        'method': 'dSPM',
        'pick_ori': None,
    })]
    assert res is stc


def test_source_derivative_path_uses_bids_derivatives(tmp_path):
    path = source_derivative_path(
        tmp_path,
        "H1",
        suffix="fwd",
        extension=".fif",
        run_id="Slow2",
        description="go",
    )

    assert path == (
        tmp_path
        / "derivatives"
        / "meg-tokens"
        / "sub-H01"
        / "meg"
        / "sub-H01_task-tokens_run-2_desc-slow-go_fwd.fif"
    )


def test_save_source_space_writes_sidecar(monkeypatch, tmp_path):
    write_src = Recorder()
    monkeypatch.setattr("meg_tokens.meg.sources.mne.write_source_spaces", write_src)
    src = object()

    out = save_source_space(src, tmp_path, "H1", "oct6")

    assert len(write_src.calls) == 1
    assert out.endswith("sub-H01_task-tokens_space-subject_desc-oct6_src.fif")
    assert os.path.exists(out.replace(".fif", ".json"))


def test_save_source_estimates_writes_manifest(tmp_path):
    class StcStub:
        def __init__(self, data):
            self.data = data
            self.save_calls = []

        def save(self, *args, **kwargs):
            self.save_calls.append((args, kwargs))

    stc1 = StcStub(np.zeros((4, 10)))
    stc2 = StcStub(np.ones((4, 10)))

    manifest = save_source_estimates([stc1, stc2], tmp_path, "H1", "Slow1", None, "go", "dSPM")

    assert len(stc1.save_calls) == 1
    assert len(stc2.save_calls) == 1
    assert stc1.save_calls[0][1]["ftype"] == "stc"
    assert stc2.save_calls[0][1]["ftype"] == "stc"
    manifest_df = pd.read_csv(manifest, sep="\t")
    assert manifest_df["trial"].tolist() == [1, 2]
    assert manifest_df["subject"].tolist() == ["H01", "H01"]
    assert manifest_df["condition"].tolist() == ["Slow", "Slow"]
    assert manifest_df["method"].tolist() == ["dSPM", "dSPM"]
