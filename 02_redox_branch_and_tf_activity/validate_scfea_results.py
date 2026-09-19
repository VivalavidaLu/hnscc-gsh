"""Check like-for-like frozen scFEA contrasts; does not run scFEA inference."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_archive import check_scfea

if __name__ == '__main__':
    print(json.dumps(check_scfea(), indent=2))
