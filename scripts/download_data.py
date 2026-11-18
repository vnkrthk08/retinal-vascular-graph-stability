from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_graph.data import ensure_hrf_data, ensure_chase_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrf-only", action="store_true")
    parser.add_argument("--require-chase", action="store_true")
    args = parser.parse_args()

    hrf_root = ensure_hrf_data(ROOT)
    print(f"Validated HRF data: {hrf_root}")

    if args.hrf_only:
        print("CHASE_DB1 skipped. HRF is ready for the notebook.")
    else:
        try:
            print(f"Validated CHASE_DB1 data: {ensure_chase_data(ROOT)}")
        except Exception as exc:
            if args.require_chase:
                raise
            print("\nWARNING: CHASE_DB1 is unavailable; HRF is ready.")
            print(str(exc))
            print("The notebook now defaults to HRF-only mode.")
