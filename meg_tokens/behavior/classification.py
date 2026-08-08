"""Trial-class assignment: a derivative-stage judgement, applied to a table.

The acquisition software records a trial's class as ``sTrialClass``, but for
a large share of trials it records ``'x'`` -- "randomly generated", with no
class asserted. Recovering a class for those trials means reading it back out
of the trial's *designed* success-probability profile
(:func:`meg_tokens.behavior.math.probability.classify_design_profile`), which
is an interpretation of the data, not a fact recorded in it. See
``docs/behavior_t0_1_nprob_trial_class.md`` section 3b.

That is why this lives apart from the TDMS parser. The parser transcribes
what LabVIEW wrote -- including ``sTrialClassRaw`` and the design profile
``sp_design_correct`` that inference consumes -- and stops there, so the raw
BIDS layer it feeds carries no assertion this project is not entitled to
make. Classification is applied afterwards, by the derivative stage that owns
the choice (``ProjectConfig.infer_random_classes``).

Because the inputs both survive into the raw table, this is re-appliable: a
derivative can be regenerated, with inference on or off, from the staged raw
behavior alone, with no access to the original ``.tdms`` container.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from meg_tokens.behavior.math.probability import classify_design_profile
from meg_tokens.behavior.schema import CLASSIFICATION_COLUMNS

# Classes the acquisition software asserts directly.
RECORDED_CLASSES: Final[dict[str, int]] = {"e": 1, "a": 2, "m": 3}


def _is_missing(value: object) -> bool:
    """Whether a design profile is absent.

    Matches ``meg_tokens.behavior.schema``'s own missing-value convention:
    ``None`` *or* NaN. A table read with plain ``pandas.read_csv`` yields NaN
    where the strict reader yields ``None``, and the schema explicitly
    permits both -- so treating only ``None`` as absent would crash on a
    table the validator had just accepted.
    """
    return value is None or (isinstance(value, float) and pd.isna(value))


def classify_trial(trial_class_raw: object, sp_design_correct: object, *, infer_random_classes: bool) -> tuple:
    """Class, source and rule for one trial's recorded label.

    Returns ``(sTrialClass, trial_class_source, trial_class_rule)``. The rule
    is recorded alongside the class so a reader can always tell *how* a trial
    came by its label -- a recorded one and an inferred one are not equally
    strong evidence, and analyses that need only the former can filter on it.
    """
    normalized = str(trial_class_raw).lower()

    if normalized in RECORDED_CLASSES:
        return RECORDED_CLASSES[normalized], "design", "recorded_label"

    if normalized == "x":
        # Checked before the profile: with inference off, this trial's class
        # is declined as a matter of policy, whether or not a profile exists.
        if not infer_random_classes:
            return 0, "unclassified", "inference_disabled"
        if _is_missing(sp_design_correct):
            # No correct target means no designed profile to read a class
            # out of -- unclassifiable in principle, not merely declined.
            return 0, "unclassified", "no_correct_target"
        trial_class, rule = classify_design_profile(sp_design_correct)
        return trial_class, ("inferred" if trial_class else "unclassified"), rule

    if normalized == "r":
        return 0, "not_applicable", "not_applicable"

    try:
        trial_class = int(trial_class_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Unrecognized sTrialClassRaw value {trial_class_raw!r}: expected "
            "'x', 'e', 'a', 'm', 'r', or an integer. The TDMS parser rejects "
            "these at transcription, so a table carrying one has been edited "
            "by hand or written by something else."
        ) from None
    source = "design" if trial_class in (1, 2, 3) else "unclassified"
    return trial_class, source, "recorded_numeric_label"


def classify_trials(table: pd.DataFrame, *, infer_random_classes: bool = True) -> pd.DataFrame:
    """Attach ``CLASSIFICATION_COLUMNS`` to a raw behavioral table.

    Reads only ``sTrialClassRaw`` and ``sp_design_correct``, both of which
    the raw layer preserves, so this can be applied to a freshly parsed table
    or to one loaded back from ``BIDS/sub-*/beh/``. The input is not modified.

    Raises ``ValueError`` if a recorded label is neither a known code nor an
    integer -- the parser has already rejected those at transcription time,
    so reaching this is a sign the table was edited by hand.
    """
    missing = [c for c in ("sTrialClassRaw", "sp_design_correct") if c not in table.columns]
    if missing:
        raise ValueError(
            f"Cannot classify trials: table is missing {missing}. Expected a raw "
            "behavioral table (meg_tokens.behavior.tdms.parse_tdms_file output, or "
            "a BIDS/sub-*/beh/*_beh.tsv read with read_raw_behavior_table)."
        )

    out = table.copy()
    assigned = []
    for position, (_, row) in enumerate(out.iterrows()):
        try:
            assigned.append(
                classify_trial(
                    row["sTrialClassRaw"],
                    row["sp_design_correct"],
                    infer_random_classes=infer_random_classes,
                )
            )
        except ValueError as error:
            # Name the offending trial: without it the caller has a value and
            # no way to find which of several hundred rows carries it.
            trial = row["nTrialIndex"] if "nTrialIndex" in out.columns else position
            raise ValueError(f"Trial {trial}: {error}") from None
    for position, column in enumerate(CLASSIFICATION_COLUMNS):
        out[column] = [values[position] for values in assigned]
    return out
