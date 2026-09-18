"""Run an original GPU diagnostic in an isolated output directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ROOT / "code/experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/reproduction")
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    allowed = {p.stem: p for p in DRIVERS.glob("e[0-9]*.py")}
    if args.list:
        print("\n".join(sorted(allowed)))
        return
    name = Path(args.experiment or "").stem
    if name not in allowed:
        parser.error("Choose an experiment shown by --list.")
    if name in ("e7_policy_bound", "e15_potential"):
        mode = args.arguments[0] if args.arguments else "all"
        if mode not in {"all", "1d", "2d", "obstacle"}:
            parser.error("Choose a mode: all, 1d, 2d, or obstacle.")
    import torch
    if not torch.cuda.is_available():
        parser.error("These diagnostic drivers require CUDA; the checks/ suite runs on CPU.")
    work = args.output_dir.resolve()
    if work == ROOT or any(work == ROOT / folder or work.is_relative_to(ROOT / folder)
                           for folder in ("results", "figures", "code", "checks", "scripts", ".git")):
        parser.error("Choose a separate output directory, for example runs/reproduction.")
    for folder in ("results", "figures"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    for checkpoint in (ROOT / "results").glob("e5_critics*.pt"):
        dest = work / "results" / checkpoint.name
        if not dest.exists():
            shutil.copy2(checkpoint, dest)
    record = {"experiment": name, "arguments": args.arguments,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    with (work / "run_history.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(record) + "\n")
    os.chdir(work)
    sys.path.insert(0, str(ROOT / "code"))
    sys.argv = [str(allowed[name]), *args.arguments]
    runpy.run_path(str(allowed[name]), run_name="__main__")


if __name__ == "__main__":
    main()
