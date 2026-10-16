import mne
import numpy as np

from meg_tokens.meg.preprocessing import HEAD_COILS
from meg_tokens.meg.sources import cleaning_operator


def test_the_cleaning_operator_is_recovered_from_the_stream_output():
    rng = np.random.default_rng(0)
    n_channels, n_removed, n_times = 30, 4, 12_000
    trunk = 1e-12 * rng.standard_normal((n_channels, n_times))
    # An oblique subtraction, as ICA and DSS make: four components removed
    # through their own patterns and filters, not an orthogonal projection.
    patterns = rng.standard_normal((n_channels, n_removed))
    filters = np.linalg.pinv(patterns) + 0.3 * rng.standard_normal((n_removed, n_channels))
    filters = np.linalg.solve(filters @ patterns, filters)
    cleaning = np.eye(n_channels) - patterns @ filters
    # Then the head-movement step takes out what the coils predict.
    coils = 1e-3 * np.cumsum(rng.standard_normal((9, n_times)), axis=1)
    movement = 1e-9 * rng.standard_normal((n_channels, 9))
    trunk_with_movement = trunk + movement @ coils
    cleaned = cleaning @ trunk_with_movement
    cleaned -= np.linalg.lstsq(coils.T, cleaned.T, rcond=None)[0].T @ coils
    names = [f"M{i:02d}" for i in range(n_channels)]
    info = mne.create_info(names + [f"{name}-2104" for name in HEAD_COILS], 600.0, ["mag"] * n_channels + ["misc"] * 9)
    as_raw = lambda mags: mne.io.RawArray(np.vstack([mags, coils]).astype(np.float32).astype(float), info, verbose=False)

    operator, record = cleaning_operator(
        as_raw(cleaned), as_raw(trunk_with_movement - cleaned), rank=n_channels - n_removed,
    )

    assert np.abs(operator - cleaning).max() < 1e-3 * np.abs(cleaning).max()
    assert record["operator_residual"] < 1e-6
    assert record["operator_rank_gap"] > 1e3


def test_an_outer_surface_is_cleared_of_the_inner_one_only_where_they_meet():
    from mne.surface import _get_ico_surface

    from meg_tokens.meg.sources import BEM_GAP_MM, _clear_of, _inside

    sphere = _get_ico_surface(4)
    inner = {"rr": 80.0 * sphere["rr"], "tris": sphere["tris"]}
    # A bump on the inner surface reaches 2 mm through the outer one.
    bump = sphere["rr"][:, 2] > 0.97
    inner["rr"][bump] *= 92.0 / 80.0
    outer = {"rr": 90.0 * sphere["rr"], "tris": sphere["tris"]}
    before = outer["rr"].copy()

    moved, shift = _clear_of(outer, inner, BEM_GAP_MM)

    assert _inside(outer, inner["rr"]).all()
    distance = np.linalg.norm(outer["rr"] - before, axis=1)
    far = sphere["rr"][:, 2] < 0.8
    assert moved > 0 and (distance[far] == 0).all()
    assert 2.0 < shift < 6.0


def test_every_deep_structure_is_named_as_freesurfer_6_names_it():
    from meg_tokens.meg.sources import VOLUME_LABELS

    lut, _ = mne.read_freesurfer_lut()
    assert set(VOLUME_LABELS) <= set(lut)


def test_a_deep_source_belongs_to_the_structure_the_aseg_puts_it_in(tmp_path):
    import nibabel

    from meg_tokens.meg.sources import deep_structures

    lut, _ = mne.read_freesurfer_lut()
    codes = np.zeros((32, 32, 32), dtype=np.int32)
    codes[:16] = lut["Brain-Stem"]
    codes[16:, :16] = lut["Left-Cerebellum-Cortex"]
    aseg = nibabel.MGHImage(codes, np.eye(4))
    (tmp_path / "S" / "mri").mkdir(parents=True)
    nibabel.save(aseg, tmp_path / "S" / "mri" / "aseg.mgz")
    to_surface_ras = aseg.header.get_vox2ras_tkr()
    voxels = np.array([[4, 4, 4], [20, 4, 4], [20, 20, 4], [8, 30, 30]])
    volume = {
        "subject_his_id": "S",
        "coord_frame": mne.io.constants.FIFF.FIFFV_COORD_MRI,
        "rr": mne.transforms.apply_trans(to_surface_ras, voxels) / 1000,
        "vertno": np.array([0, 1, 3]),
    }

    found = deep_structures([volume], tmp_path)

    assert found["Brain-Stem"].tolist() == [0, 2]
    assert found["Left-Cerebellum-Cortex"].tolist() == [1]
    assert found["Right-Cerebellum-Cortex"].size == 0
