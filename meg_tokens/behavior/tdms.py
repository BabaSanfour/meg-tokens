"""Parse raw Tokens-task TDMS files into canonical Stage 1 trial fields.

This is the only behavior module that knows the LabVIEW ``Events`` text format
or TDMS group layout. It preserves recorded trial content, derives designed
success-probability profiles and auditable class labels, and validates the
result before it crosses into the table layer.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from nptdms import TdmsFile

from meg_tokens.behavior.math.probability import (
    classify_design_profile,
    success_probability_profile,
)
from meg_tokens.behavior.schema import (
    TRIAL_COLUMNS,
    parse_token_directions,
    validate_behavior_table,
)
from meg_tokens.core import normalize_subject_id


FILENAME_RE = re.compile(
    r"^(?P<subject>H(?:0[1-9]|[1-9][0-9]*))"
    r"(?P<condition>Slow|Fast|RT)(?P<run>[0-9]+)_(?P<date>[0-9]{6})\.tdms$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TdmsRunInfo:
    """Canonical entities encoded in one behavioral TDMS filename.

    Attributes
    ----------
    subject
        Normalized subject identifier without a ``sub-`` prefix.
    condition
        ``Fast``, ``Slow``, or ``RT``.
    run
        Run number preserved as filename text.
    date
        Six-digit acquisition date preserved as filename text.
    """

    subject: str
    condition: str
    run: str
    date: str


def parse_tdms_filename(filename: str) -> TdmsRunInfo:
    """Parse a project TDMS basename into canonical run entities.

    Parameters
    ----------
    filename
        Basename matching ``H<subject><condition><run>_<YYMMDD>.tdms``. Paths
        and non-TDMS suffixes are not accepted.

    Returns
    -------
    TdmsRunInfo
        Normalized subject and condition plus the encoded run and date.

    Raises
    ------
    ValueError
        If the basename does not match the project naming convention.
    """
    match = FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Non-standard TDMS filename: {filename}")
    raw_condition = match.group("condition")
    # "RT".capitalize() would produce "Rt"; RT is an initialism, not a word.
    return TdmsRunInfo(
        subject=normalize_subject_id(match.group("subject")),
        condition="RT" if raw_condition.upper() == "RT" else raw_condition.capitalize(),
        run=match.group("run"),
        date=match.group("date"),
    )


def parse_single_trial(events_str: str, *, infer_random_classes: bool = True) -> dict:
    """Parse the ``Events`` property from one TDMS trial group.

    Parameters
    ----------
    events_str
        Raw LabVIEW event block containing scalar trial metadata and an
        optional ``Tokens.Data`` section.
    infer_random_classes
        If true, trials recorded as ``sTrialClass='x'`` are classified from
        their complete correct-target design profile. If false, they remain
        class ``0``. Recorded ``e``, ``a``, and ``m`` labels are always
        preserved.

    Returns
    -------
    dict
        One row of canonical ``TRIAL_COLUMNS`` values. Runtime token fields are
        parallel Python lists; timestamps are milliseconds; probabilities are
        constrained to ``[0, 1]``; class provenance and rule are explicit.

    Raises
    ------
    ValueError
        If required structural metadata is missing, the designed sequence or
        class label is invalid, token-log fields have unequal lengths, or a
        logged probability lies outside the accepted floating-point tolerance.

    Notes:
        LabVIEW writes ``nProb`` in whichever notation it pleases: most rows
        are plain decimals, but values near 0 and 1 are emitted in scientific
        notation (for example, ``5.46875E-1`` and
        ``-2.22044604925031E-16``). A digits-and-dots pattern silently drops
        both the sign and the exponent, turning ``5.46875E-1`` into
        ``5.46875``—an out-of-range success probability that then corrupts
        trial classification.

        The parser builds ``nTokenNum``, ``nTokenDir``, ``tTime``, and
        ``nProb`` independently from the same ``Tokens.Data`` rows. If a row
        matches only some patterns, the arrays become misaligned and every
        subsequent index-based criterion is corrupted, so their lengths must
        match.

        Because ``nProb`` is a probability, values outside ``[0, 1]`` indicate a
        parsing error rather than a real experimental outcome. Validation uses
        a small tolerance for LabVIEW's floating-point representation of zero,
        which it may write as ``-2.22044604925031E-16``.

        The parser does not synthesize missing token rows or event timestamps.
        Response-dependent scalar timestamps may be absent and are represented
        as zero, while structural fields are mandatory.
    """
    # Normalize line endings and split
    lines = [line.strip() for line in events_str.replace('\r\n', '\n').split('\n') if line.strip()]
    
    meta = {}
    n_token_num_list = []
    n_token_dir_list = []
    t_time_list = []
    n_prob_list = []
    
    # Regular expressions for parsing token steps.
    token_num_re = re.compile(r'nTokenNum:\s*(\d+)')
    token_dir_re = re.compile(r'nTokenDir:\s*(\d+)')
    time_re = re.compile(r'tTime:\s*(\d+)')
    prob_re = re.compile(r'nProb:\s*(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)')

    in_tokens = False
    for line in lines:
        if line.startswith("Tokens.Data:"):
            in_tokens = True
            continue
        if in_tokens:
            if line == ")":
                in_tokens = False
                continue
            # Parse step metrics
            token_num_match = token_num_re.search(line)
            token_dir_match = token_dir_re.search(line)
            time_match = time_re.search(line)
            prob_match = prob_re.search(line)
            if token_num_match:
                n_token_num_list.append(int(token_num_match.group(1)))
            if token_dir_match:
                n_token_dir_list.append(int(token_dir_match.group(1)))
            if time_match:
                t_time_list.append(int(time_match.group(1)))
            if prob_match:
                n_prob_list.append(float(prob_match.group(1)))
        else:
            if ":" in line:
                k, v = line.split(":", 1)
                # Strip quotes from value
                val = v.strip().strip("'").strip('"')
                meta[k.strip()] = val

    # These fields describe trial structure that exists regardless of
    # whether the subject responded (trial index, go cue, trial end, the
    # predetermined correct target, the token sequence). Their absence means
    # this trial's Events block is truncated or corrupted, not a real
    # experimental outcome, so it is a hard error rather than a 0 default.
    required_fields = (
        'nTrialIndex', 'nInitialTime', 'nCorrectChoice', 'tGO', 'tTrialEnd',
        'sTokenDirs', 'nOutcome',
    )
    missing_fields = [key for key in required_fields if key not in meta]
    if missing_fields:
        trial_hint = meta.get('nTrialIndex', '?')
        raise ValueError(
            f"Trial {trial_hint}: Events block is missing required "
            f"field(s) {missing_fields}"
        )

    # Extract base variables
    n_trial_index = int(meta['nTrialIndex'])
    n_initial_time = int(meta['nInitialTime'])
    n_choice_made = int(meta.get('nChoiceMade', 0))
    n_correct_choice = int(meta['nCorrectChoice'])

    n_outcome = int(meta['nOutcome'])
    t_go = int(meta['tGO'])
    # tEnterTarget only exists once a target has actually been entered, so
    # it is legitimately absent (not corrupted) on a skipped trial. The same
    # holds for the center-hold timestamps. tExitCenter is retained because it
    # is the only field that could carry a movement duration
    # (tEnterTarget - tExitCenter); see docs/behavior_qc_report.md for what it
    # actually contains in this dataset.
    t_enter_center = int(meta.get('tEnterCenter', 0))
    t_exit_center = int(meta.get('tExitCenter', 0))
    t_enter_target = int(meta.get('tEnterTarget', 0))
    t_trial_end = int(meta['tTrialEnd'])

    # sTokenDirs is a string representation of token directions
    s_token_dirs = meta['sTokenDirs']

    if n_correct_choice in (1, 2):
        try:
            design_directions = parse_token_directions(s_token_dirs)
            sp_design_correct = success_probability_profile(
                design_directions,
                target=n_correct_choice,
            )
        except ValueError as error:
            raise ValueError(
                f"Trial {n_trial_index}: invalid designed token sequence: {error}"
            ) from error
    else:
        sp_design_correct = None

    # Preserve recorded designed labels. Only random ('x') trials are assigned
    # a posteriori from the complete, correct-target designed SP profile.
    trial_class_raw = meta.get('sTrialClass', 'x')
    normalized_trial_class = str(trial_class_raw).lower()
    recorded_classes = {'e': 1, 'a': 2, 'm': 3}
    if normalized_trial_class in recorded_classes:
        s_trial_class = recorded_classes[normalized_trial_class]
        trial_class_source = 'design'
        trial_class_rule = 'recorded_label'
    elif normalized_trial_class == 'x':
        if not infer_random_classes:
            s_trial_class = 0
            trial_class_source = 'unclassified'
            trial_class_rule = 'inference_disabled'
        elif sp_design_correct is None:
            s_trial_class = 0
            trial_class_source = 'unclassified'
            trial_class_rule = 'no_correct_target'
        else:
            s_trial_class, trial_class_rule = classify_design_profile(
                sp_design_correct
            )
            trial_class_source = 'inferred' if s_trial_class else 'unclassified'
    elif normalized_trial_class == 'r':
        s_trial_class = 0
        trial_class_source = 'not_applicable'
        trial_class_rule = 'not_applicable'
    else:
        try:
            s_trial_class = int(trial_class_raw)
        except ValueError:
            raise ValueError(
                f"Trial {n_trial_index}: unrecognized sTrialClass value "
                f"{trial_class_raw!r} (expected 'x', 'e', 'a', 'm', or an "
                "integer)"
            )
        trial_class_source = 'design' if s_trial_class in (1, 2, 3) else 'unclassified'
        trial_class_rule = 'recorded_numeric_label'

    token_log_rows = len(n_prob_list)
    token_log_short = token_log_rows == 14

    # Preserve every recorded runtime token row, including on no-choice trials.
    n_token_num = n_token_num_list
    n_token_dir = n_token_dir_list
    t_time = t_time_list
    n_prob = n_prob_list

    token_field_lengths = {
        'nTokenNum': len(n_token_num),
        'nTokenDir': len(n_token_dir),
        'tTime': len(t_time),
        'nProb': len(n_prob),
    }
    if len(set(token_field_lengths.values())) != 1:
        raise ValueError(
            f"Trial {n_trial_index}: Tokens.Data field lengths differ: "
            f"{token_field_lengths}"
        )
    out_of_range = [value for value in n_prob if not -1e-9 <= value <= 1 + 1e-9]
    if out_of_range:
        raise ValueError(
            f"Trial {n_trial_index}: nProb values outside [0, 1]: "
            f"{out_of_range}"
        )
    n_prob = [min(1.0, max(0.0, value)) for value in n_prob]

    return {
        'nTrialIndex': n_trial_index,
        'sTrialClass': s_trial_class,
        'sTrialClassRaw': trial_class_raw,
        'trial_class_source': trial_class_source,
        'trial_class_rule': trial_class_rule,
        'sp_design_correct': sp_design_correct,
        'nInitialTime': n_initial_time,
        'nChoiceMade': n_choice_made,
        'nCorrectChoice': n_correct_choice,
        'tGO': t_go,
        'tEnterCenter': t_enter_center,
        'tExitCenter': t_exit_center,
        'tEnterTarget': t_enter_target,
        'tTrialEnd': t_trial_end,
        'sTokenDirs': s_token_dirs,
        'nTokenNum': n_token_num,
        'nTokenDir': n_token_dir,
        'tTime': t_time,
        'nProb': n_prob,
        'token_log_rows': token_log_rows,
        'token_log_short': token_log_short,
        'nOutcome': n_outcome,
    }

def parse_tdms_file(
    file_path: str,
    *,
    infer_random_classes: bool = True,
) -> pd.DataFrame:
    """Parse and validate every trial in one behavioral TDMS file.

    Parameters
    ----------
    file_path
        Path to the source TDMS file.
    infer_random_classes
        Passed unchanged to :func:`parse_single_trial`.

    Returns
    -------
    pandas.DataFrame
        Canonical trial rows ordered as encountered in the file and restricted
        to ``TRIAL_COLUMNS``. Run identity is added later by
        :func:`meg_tokens.behavior.tables.add_run_metadata`.

    Raises
    ------
    ValueError
        If a ``TrialData`` group lacks its ``Events`` property, no trial data
        exist, an individual event block is invalid, or final Stage 1
        validation fails. Errors include the source path and trial context.
    """
    tdms_file = TdmsFile(file_path)

    trial_data = []
    missing_events = []
    for group in tdms_file.groups():
        if not group.name.startswith("TrialData"):
            continue
        events_str = group.properties.get('Events', '')
        if not events_str:
            missing_events.append(group.name)
            continue
        try:
            trial_dict = parse_single_trial(
                events_str,
                infer_random_classes=infer_random_classes,
            )
        except ValueError as error:
            raise ValueError(f"{file_path}: {error}") from error
        trial_data.append(trial_dict)

    if missing_events:
        raise ValueError(
            f"{file_path}: TrialData group(s) with no 'Events' property: "
            f"{missing_events}"
        )
    if not trial_data:
        raise ValueError(f"{file_path}: no TrialData groups with trial data found")

    df = pd.DataFrame(trial_data, columns=TRIAL_COLUMNS)
    try:
        validate_behavior_table(df)
    except ValueError as error:
        raise ValueError(f"{file_path}: {error}") from error
    return df
