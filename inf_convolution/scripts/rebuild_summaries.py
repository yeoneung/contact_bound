"""Rebuild numerical summaries in a self-contained extracted package."""
from pathlib import Path
from make_artifacts import main

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    for name in ['documents', 'research', 'figures', 'manuscript/tables', 'manuscript/sections']:
        (root / name).mkdir(parents=True, exist_ok=True)
    main()
