"""TDMS parsing and validation for Tokens behavioral logs."""

import re
from dataclasses import dataclass

import pandas as pd
from nptdms import TdmsFile

from meg_tokens.core import normalize_subject_id


TRIAL_COLUMNS = [
    'nTrialIndex', 'sTrialClass', 'nInitialTime', 'nChoiceMade',
    'nCorrectChoice', 'tGO', 'tEnterTarget', 'tTrialEnd',
    'sTokenDirs', 'tTime', 'nProb'
]

FILENAME_RE = re.compile(
    r"^(?P<subject>H(?:0[1-9]|[1-9][0-9]*))"
    r"(?P<condition>Slow|Fast|RT)(?P<run>[0-9]+)_(?P<date>.*)\.tdms$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TdmsRunInfo:
    subject: str
    condition: str
    run: str
    date: str


def parse_tdms_filename(filename: str) -> TdmsRunInfo:
    """Parse one project TDMS filename into canonical run entities."""
    match = FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Non-standard TDMS filename: {filename}")
    raw_condition = match.group("condition")
    return TdmsRunInfo(
        subject=normalize_subject_id(match.group("subject")),
        condition="RT" if raw_condition.upper() == "RT" else raw_condition.capitalize(),
        run=match.group("run"),
        date=match.group("date"),
    )


def add_run_metadata(
    table: pd.DataFrame,
    run_info: TdmsRunInfo,
    source_file: str,
) -> pd.DataFrame:
    """Add canonical run identity and derived response fields."""
    out = table.copy()
    out.insert(0, "subject", run_info.subject)
    out.insert(1, "condition", run_info.condition)
    out.insert(2, "run", int(run_info.run))
    out.insert(3, "source_file", source_file)
    out["rawRT"] = (out["tEnterTarget"] - out["tGO"]).where(out["nChoiceMade"] > 0)
    out["isCorrect"] = (
        out["nChoiceMade"] == out["nCorrectChoice"]
    ).where(out["nChoiceMade"] > 0)
    return out


def validate_behavior_dataframe(df: pd.DataFrame) -> None:
    """Validate columns and basic event timing for a parsed behavior table."""
    missing = [col for col in TRIAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Behavior table is missing required columns: {missing}")

    if not df.empty:
        expected_index = list(range(1, len(df) + 1))
        actual_index = df['nTrialIndex'].astype(int).tolist()
        if actual_index != expected_index:
            raise ValueError("nTrialIndex must be sequential and start at 1")

    chosen = df['nChoiceMade'] > 0
    invalid_order = chosen & (
        (df['tGO'] > df['tEnterTarget']) |
        (df['tEnterTarget'] > df['tTrialEnd'])
    )
    if invalid_order.any():
        rows = df.loc[invalid_order, 'nTrialIndex'].tolist()
        raise ValueError(f"Invalid event ordering for trials: {rows}")

def parse_single_trial(events_str: str) -> dict:
    """
    Parses the Events property string from a single trial group.
    
    Args:
        events_str: The raw semicolon or CRLF-separated events string.
        
    Returns:
        A dictionary of parsed trial metrics.
    """
    # Normalize line endings and split
    lines = [line.strip() for line in events_str.replace('\r\n', '\n').split('\n') if line.strip()]
    
    meta = {}
    t_time_list = []
    n_prob_list = []
    
    # Regular expressions for parsing token steps
    time_re = re.compile(r'tTime:\s*(\d+)')
    prob_re = re.compile(r'nProb:\s*([\d\.]+)')
    
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
            time_match = time_re.search(line)
            prob_match = prob_re.search(line)
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

    # Extract base variables
    n_trial_index = int(meta.get('nTrialIndex', 0))
    n_initial_time = int(meta.get('nInitialTime', 0))
    n_choice_made = int(meta.get('nChoiceMade', 0))
    n_correct_choice = int(meta.get('nCorrectChoice', 0))
    
    t_go = int(meta.get('tGO', 0))
    t_enter_target = int(meta.get('tEnterTarget', 0))
    t_trial_end = int(meta.get('tTrialEnd', 0))
    
    # sTokenDirs is a string representation of token directions
    s_token_dirs = meta.get('sTokenDirs', '0')
    
    # Parse and map sTrialClass
    trial_class_raw = meta.get('sTrialClass', 'x')
    if trial_class_raw == 'x':
        s_trial_class = 0
    elif trial_class_raw == 'e':
        s_trial_class = 1
    elif trial_class_raw == 'a':
        s_trial_class = 2
    elif trial_class_raw == 'm':
        s_trial_class = 3
    else:
        try:
            s_trial_class = int(trial_class_raw)
        except ValueError:
            s_trial_class = 0

    # Handle token time series arrays
    if n_choice_made == 0:
        t_time = 0
        n_prob = 0
    else:
        t_time = t_time_list
        n_prob = n_prob_list
        
        # Replicate DDM logic rules for sTrialClass override
        if len(n_prob) >= 8:
            if n_prob[1] > 0.6 and n_prob[4] > 0.75 and n_prob[7] > 0.75:
                s_trial_class = 1 # 1 = Easy
            elif n_prob[1] == 0.5 and n_prob[2] > 0.38 and n_prob[2] < 0.65 and n_prob[4] > 0.35 and n_prob[4] < 0.65:
                s_trial_class = 2 # 2 = Ambiguous
            elif n_prob[2] < 0.4:
                s_trial_class = 3 # 3 = Misleading (Fixed bug from previous member codebase)

    return {
        'nTrialIndex': n_trial_index,
        'sTrialClass': s_trial_class,
        'nInitialTime': n_initial_time,
        'nChoiceMade': n_choice_made,
        'nCorrectChoice': n_correct_choice,
        'tGO': t_go,
        'tEnterTarget': t_enter_target,
        'tTrialEnd': t_trial_end,
        'sTokenDirs': s_token_dirs,
        'tTime': t_time,
        'nProb': n_prob
    }

def parse_tdms_file(file_path: str) -> pd.DataFrame:
    """
    Parses a complete TDMS file and extracts behavioral trial variables.
    
    Args:
        file_path: Absolute path to the TDMS file.
        
    Returns:
        A pandas DataFrame containing parsed trial data.
    """
    tdms_file = TdmsFile(file_path)
    
    trial_data = []
    for group in tdms_file.groups():
        if group.name.startswith("TrialData"):
            events_str = group.properties.get('Events', '')
            if events_str:
                trial_dict = parse_single_trial(events_str)
                trial_data.append(trial_dict)
                
    df = pd.DataFrame(trial_data, columns=TRIAL_COLUMNS)
    validate_behavior_dataframe(df)
    return df
