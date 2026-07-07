import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from meg_tokens.io import derivative_path, save_table
from meg_tokens.utils.tdms_parser import parse_tdms_file, validate_behavior_dataframe

# Regex to parse DDM TDMS filenames like H1Slow1_180131.tdms
FILENAME_RE = re.compile(
    r'^(?P<subject>H(?:0[1-9]|[1-9][0-9]*))(?P<condition>Slow|Fast|RT)(?P<run>[0-9]+)_(?P<date>.*)\.tdms$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TdmsRunInfo:
    subject: str
    condition: str
    run: str
    date: str


def normalize_subject_id(subject_id: str) -> str:
    """Normalize legacy subject labels to H01-style IDs."""
    match = re.fullmatch(r'H0*([0-9]+)', subject_id.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid subject ID: {subject_id}")
    return f"H{int(match.group(1)):02d}"


def parse_tdms_filename(filename: str) -> TdmsRunInfo:
    match = FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Non-standard TDMS filename: {filename}")
    return TdmsRunInfo(
        subject=normalize_subject_id(match.group("subject")),
        condition="RT" if match.group("condition").upper() == "RT" else match.group("condition").capitalize(),
        run=match.group("run"),
        date=match.group("date"),
    )


def behavior_output_path(output_root: str, run_info: TdmsRunInfo) -> Path:
    return derivative_path(
        output_root,
        subject=run_info.subject,
        datatype="beh",
        task="tokens",
        run=run_info.run,
        description=run_info.condition.lower(),
        suffix="beh",
        extension=".tsv",
    )


def add_run_metadata(df: pd.DataFrame, run_info: TdmsRunInfo, source_file: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "subject", run_info.subject)
    out.insert(1, "condition", run_info.condition)
    out.insert(2, "run", int(run_info.run))
    out.insert(3, "source_file", source_file)
    out["rawRT"] = (out["tEnterTarget"] - out["tGO"]).where(out["nChoiceMade"] > 0)
    out["isCorrect"] = (out["nChoiceMade"] == out["nCorrectChoice"]).where(out["nChoiceMade"] > 0)
    return out

def process_subject_tdms(subject_id: str, input_dir: str, output_dir: str, dry_run: bool = False) -> list:
    """
    Scans a subject's TDMS directory, parses all trial runs, and saves them
    as BIDS-derivatives-style behavior `.tsv` tables with JSON sidecars.
    
    Args:
        subject_id: Subject directory name (e.g. 'H1', 'H02').
        input_dir: Path to the main tdms/ folder (containing subject subfolders).
        output_dir: Path to the target dataframes/ folder.
        dry_run: If True, only prints mapping and does not write files.
        
    Returns:
        A list of dicts with mapped files.
    """
    subject_input_path = os.path.join(input_dir, subject_id)
    
    if not os.path.exists(subject_input_path):
        raise FileNotFoundError(f"Subject input directory does not exist: {subject_input_path}")
        
    processed_files = []
    
    # List all TDMS files in subject directory
    for filename in sorted(os.listdir(subject_input_path)):
        if filename.endswith(".tdms"):
            try:
                run_info = parse_tdms_filename(filename)
                input_file_path = os.path.join(subject_input_path, filename)
                output_file_path = behavior_output_path(output_dir, run_info)

                print(f"Mapping: {filename} -> {output_file_path}")
                
                if not dry_run:
                    df = parse_tdms_file(input_file_path)
                    df = add_run_metadata(df, run_info, filename)
                    validate_behavior_dataframe(df)
                    save_table(
                        output_file_path,
                        df,
                        metadata={
                            "stage": "behavior_parsing",
                            "subject": run_info.subject,
                            "condition": run_info.condition,
                            "run": run_info.run,
                            "source_file": filename,
                            "source_date": run_info.date,
                        },
                    )
                else:
                    df = pd.DataFrame()
                    
                processed_files.append({
                    'subject': run_info.subject,
                    'condition': run_info.condition,
                    'run': run_info.run,
                    'input': filename,
                    'output': str(output_file_path),
                    'trials': len(df) if not dry_run else 0
                })
            except ValueError:
                print(f"Skipping non-standard TDMS file: {filename}")
                
    return processed_files

def run_tdms_batch_processor(input_dir: str, output_dir: str, dry_run: bool = False) -> list:
    """
    Discovers all subjects under input_dir and runs batch processing.
    
    Args:
        input_dir: Path to the main tdms/ folder.
        output_dir: Path to the target dataframes/ folder.
        dry_run: If True, does not write files.
        
    Returns:
        Summary statistics of all processed files.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"TDMS base directory does not exist: {input_dir}")
        
    subjects = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("H")
    ]
    
    summary = []
    for subject_id in sorted(subjects):
        print(f"\n=== Processing Subject: {subject_id} ===")
        try:
            results = process_subject_tdms(subject_id, input_dir, output_dir, dry_run=dry_run)
            summary.extend(results)
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse TDMS files into BIDS-derivatives-style behavior TSV tables."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the main tdms/ folder (containing subject subfolders).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="BIDS derivatives root for parsed behavior tables.")
    parser.add_argument("--dry_run", action='store_true',
                        help="If set, only prints mapping and does not write files.")
    
    args = parser.parse_args()
    
    run_tdms_batch_processor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )
