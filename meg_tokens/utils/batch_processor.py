import os
import re
import pandas as pd
from meg_tokens.utils.tdms_parser import parse_tdms_file

# Regex to parse DDM TDMS filenames like H1Slow1_180131.tdms
FILENAME_RE = re.compile(r'^H(?:0[1-9]|[1-9][0-9]*)(Slow|Fast|RT)([0-9]+)_(.*)\.tdms$', re.IGNORECASE)

def process_subject_tdms(subject_id: str, input_dir: str, output_dir: str, dry_run: bool = False) -> list:
    """
    Scans a subject's TDMS directory, parses all trial runs, and saves them
    in the new clean CSV format: {subject_id}_{condition}{index}.csv.
    
    Args:
        subject_id: Subject directory name (e.g. 'H1', 'H02').
        input_dir: Path to the main tdms/ folder (containing subject subfolders).
        output_dir: Path to the target dataframes/ folder.
        dry_run: If True, only prints mapping and does not write files.
        
    Returns:
        A list of dicts with mapped files.
    """
    subject_input_path = os.path.join(input_dir, subject_id)
    subject_output_path = os.path.join(output_dir, subject_id)
    
    if not os.path.exists(subject_input_path):
        raise FileNotFoundError(f"Subject input directory does not exist: {subject_input_path}")
        
    if not dry_run:
        os.makedirs(subject_output_path, exist_ok=True)
        
    processed_files = []
    
    # List all TDMS files in subject directory
    for filename in sorted(os.listdir(subject_input_path)):
        if filename.endswith(".tdms"):
            match = FILENAME_RE.match(filename)
            if match:
                condition, index, date = match.groups()
                # Construct output file name (e.g. H1_Slow1.csv or H1_RT1.csv)
                cond_formatted = "RT" if condition.upper() == "RT" else condition.capitalize()
                output_filename = f"{subject_id}_{cond_formatted}{index}.csv"
                
                input_file_path = os.path.join(subject_input_path, filename)
                output_file_path = os.path.join(subject_output_path, output_filename)
                
                print(f"Mapping: {filename} -> {output_filename}")
                
                if not dry_run:
                    df = parse_tdms_file(input_file_path)
                    df.to_csv(output_file_path, index=True) # Replicate original index output
                    
                processed_files.append({
                    'subject': subject_id,
                    'input': filename,
                    'output': output_filename,
                    'trials': len(df) if not dry_run else 0
                })
            else:
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
        description="Run batch processor to parse TDMS files and map them to CSV dataframes."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the main tdms/ folder (containing subject subfolders).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the target dataframes/ folder.")
    parser.add_argument("--dry_run", action='store_true',
                        help="If set, only prints mapping and does not write files.")
    
    args = parser.parse_args()
    
    run_tdms_batch_processor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )
