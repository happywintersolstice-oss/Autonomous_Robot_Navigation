"""Export actual historical ZIPs, enumerate coverage, or verify sources/ZIPs."""

import argparse, json
from pathlib import Path
from archive_tools import export, verify_zip, verify_sources, enumerate_exports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-sources", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.verify_sources:
        r = verify_sources(root)
    elif a.list:
        r = {
            k: {"files": len(v), "uncompressed_bytes": sum(x["bytes"] for x in v)}
            for k, v in enumerate_exports(root).items()
        }
    elif a.verify:
        if a.output_dir is None:
            p.error("--verify requires --output-dir")
        r = verify_zip(a.output_dir)
    else:
        if a.output_dir is None:
            p.error("Provide --output-dir, --list or --verify-sources")
        i = export(root, a.output_dir)
        r = i["archives"]
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
