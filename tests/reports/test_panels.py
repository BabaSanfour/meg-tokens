import numpy as np

from meg_tokens.reports import panels


def test_within_subject_error_matches_hand_computed_value():
    # 3 subjects x 2 conditions; Cousineau-Morey normalisation by hand.
    values = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    subject_means = values.mean(axis=1, keepdims=True)
    grand_mean = values.mean()
    normalized = values - subject_means + grand_mean
    correction = np.sqrt(2 / 1)
    expected_sd = normalized.std(axis=0, ddof=1) * correction
    expected_sem = expected_sd / np.sqrt(3)

    result = panels.within_subject_error(values, kind="sem")
    np.testing.assert_allclose(result, expected_sem)
