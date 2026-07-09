import numpy as np
import pytest

from meg_tokens.analysis.decomposition import apply_neural_transform, fit_condition_pca


def test_fit_condition_pca_projects_raw_condition_means_like_nmdata():
    means = np.array([
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
    ])

    result = fit_condition_pca(means, n_components=1)

    np.testing.assert_allclose(np.abs(result.loadings[:, 0]), [1.0, 0.0])
    np.testing.assert_allclose(np.abs(result.trajectory[:, 0, :]), means[:, 0, :])
    assert result.fit_scores.shape == (6, 1)
    assert result.variance_ratio[0] == pytest.approx(1.0)


def test_fit_condition_pca_preserves_nan_padded_trajectory_times():
    means = np.array([
        [[1.0, 2.0, 3.0, np.nan], [2.0, 2.0, 2.0, np.nan]],
        [[4.0, 5.0, 6.0, np.nan], [2.0, 2.0, 2.0, np.nan]],
    ])

    result = fit_condition_pca(means, n_components=2)

    assert result.fit_scores.shape[0] == 6
    assert np.all(np.isnan(result.trajectory[:, :, -1]))
    assert result.fit_time_mask.tolist() == [True, True, True, True]


def test_apply_neural_transform_rejects_sqrt_on_signed_data():
    with pytest.raises(ValueError, match="non-negative"):
        apply_neural_transform(np.array([-1.0, 4.0]), "sqrt")

    np.testing.assert_allclose(apply_neural_transform(np.array([-1.0, 4.0]), "signed-sqrt"), [-1.0, 2.0])
