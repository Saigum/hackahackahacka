from pathlib import Path
import time,subprocess,json,csv
R=Path(__file__).resolve().parents[1]
while not (R/'output/candidate_pairs.tsv').exists():time.sleep(3)
# Wait for retrieval's resident index to be released before the official in-memory validator.
while True:
    active=False
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'work/test_pairs' in args:active=True
    if not active:break
    time.sleep(3)
cmd=['python3',str(R/'dataset/student_resource/utils/validate_submission.py'),
 '--matching',str(R/'output/matching_results.tsv'),'--candidate',str(R/'output/candidate_pairs.tsv'),
 '--test-dir',str(R/'dataset/student_resource/dataset/test'),'--check-ids']
p=subprocess.run(cmd)
raise SystemExit(p.returncode)
