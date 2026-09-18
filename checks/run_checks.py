"""Run the independent checks, optionally including saved-result CPU replays."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    commands = [[str(p)] for p in sorted((ROOT / "checks").glob("check_*.py"))]
    commands.append([str(ROOT / "code/experiments/complete_contact_certificate.py"), "--check-only"])
    if args.replay:
        commands.extend([str(p)] for p in sorted((ROOT / "checks").glob("reproduce_*_subset.py")))
    for command in commands:
        print(f"Running {Path(command[0]).name}", flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    print(f"Passed {len(commands)} check/replay programs.", flush=True)


if __name__ == "__main__":
    main()
