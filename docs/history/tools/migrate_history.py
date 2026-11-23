"""One migration: inventory unique historical bytes, archive first, verify all."""

import csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from archive_tools import verify_sources

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("C:/Users/User/Desktop/Autonomous_Robot_Navigation")
APP = "autonomous-robot-navigation-rl"
KEEP = {
    "__init__.py",
    "core.py",
    "release.py",
    "partial_agents.py",
    "partial_baselines.py",
    "partial_benchmark.py",
    "partial_contract.py",
    "partial_dynamic_env.py",
    "partial_evaluation.py",
    "partial_features.py",
    "partial_features_scaled.py",
    "partial_memory.py",
    "partial_neural.py",
    "partial_observation.py",
    "partial_sensor.py",
}


def git(root, *args, raw=False):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "-c",
            "safe.directory=" + str(root),
            *args,
        ],
        cwd=root,
        text=not raw,
    )


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def category(name):
    if (
        "/src/" in name
        or "/developer/" in name
        or name.endswith((".py", ".cmd", ".toml", ".txt"))
        or name in ("README.md", ".gitignore", ".gitattributes")
        or name.startswith("docs/")
    ):
        return "historical-source", "both"
    if "partial" in name.lower() or "CODEX_PHASE" in name or "CODEX_ROBOT" in name:
        return "partial-research", "partial"
    return "legacy-classic", "classic"


def backing(name):
    cat, _ = category(name)
    if (
        name.startswith(APP + "/artifacts/partial-dynamic-navigation-v1/")
        or name.startswith(APP + "/data/partial/")
        or name.startswith("CODEX_")
        or name.startswith(APP + "/docs/partial_")
        or name.startswith(APP + "/configs/partial_")
        or name.startswith(APP + "/src/robot_navigation/partial_")
    ):
        return name, "retain"
    if name.startswith(APP + "/src/robot_navigation/"):
        return "archive/historical-source/" + name, (
            "copy" if Path(name).name in KEEP else "move"
        )
    if name in (
        "README.md",
        "run_app.py",
        "Start Robot Navigation.cmd",
        APP + "/README.md",
        APP + "/pyproject.toml",
        APP + "/.streamlit/config.toml",
    ) or name.startswith("docs/"):
        return "archive/historical-source/" + name, "copy"
    if name.startswith(APP + "/developer/") or name.startswith(APP + "/docs/"):
        return "archive/historical-source/" + name, "move"
    if (
        name.startswith(APP + "/artifacts/")
        or name.startswith(APP + "/data/")
        or name.startswith(APP + "/configs/")
    ):
        return "archive/legacy-classic/" + name, "move"
    return "archive/historical-source/" + name, "copy"


def safe_dest(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT / "archive"):
        raise ValueError("Archive target escaped workspace: " + name)
    if p.name in (".gitattributes", ".gitignore"):
        p = p.with_name(p.name + ".archived")
    return p


def main():
    archive = ROOT / "archive"
    (archive / "manifests").mkdir(parents=True, exist_ok=False)
    (archive / ".gitattributes").write_bytes(b"* -text\n")
    entries = {}
    blob_map = {}
    actions = []
    refs = git(
        ROOT, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads"
    ).splitlines()
    manifest = {
        "schema": "complete-robot-history-v1",
        "snapshot_parent": git(ROOT, "rev-parse", "HEAD").strip(),
        "source_refs": refs,
        "files": [],
        "notes": [
            "All reachable Git blob versions plus actual original workspace and historical runtime outputs. Same-content origins share one backing file.",
            "Existing sealed partial evidence stays at original paths. Archived Git control files have .archived suffix on disk, original names inside ZIPs.",
            "Historical source is reference-only, not importable by the active app. Restore the original directory context or original commit for reproduction.",
        ],
    }

    def ingest(
        path,
        name,
        version,
        blob=None,
        destination=None,
        operation=None,
        zip_prefix="original",
    ):
        stat = path.stat()
        h = hashlib.sha256()
        b = hashlib.sha1(b"blob " + str(stat.st_size).encode() + b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
                b.update(chunk)
        digest = h.hexdigest()
        origin = {
            "path": name,
            "source_version": version,
            "git_blob": blob,
            "representation": "on-disk",
        }
        if digest in entries:
            row = entries[digest]
            row["origins"].append(origin)
            blob_map.setdefault(b.hexdigest(), row)
            if path.is_relative_to(ROOT) and operation is None:
                target, op = backing(name)
                if op in ("copy", "move"):
                    actions.append((path, safe_dest(target), op))
            return row
        cat, collection = category(name)
        if destination is None:
            destination, operation = backing(name)
        dest = ROOT / destination if operation == "retain" else safe_dest(destination)
        relative = dest.relative_to(ROOT).as_posix()
        row = {
            "original_path": name,
            "archive_path": relative,
            "bytes": stat.st_size,
            "sha256": digest,
            "git_blob": blob,
            "category": cat,
            "collection": collection,
            "source_version": version,
            "origins": [origin],
            "zip_path": zip_prefix + "/" + name,
            "preservation": operation,
        }
        # Copies from original user/worktree are read-only; never move them.
        if operation == "external-copy":
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        elif operation in ("copy", "move"):
            actions.append((path, dest, operation))
        entries[digest] = row
        blob_map[b.hexdigest()] = row
        return row

    tracked = git(ROOT, "ls-files", "--stage", "-z").split("\0")[:-1]
    for ix, line in enumerate(tracked):
        header, name = line.split("\t", 1)
        blob = header.split()[1]
        ingest(ROOT / name, name, manifest["snapshot_parent"], blob)
        if ix % 2000 == 0:
            print("Inventory tracked", ix, flush=True)
    # Original dirty workspace: all tracked/unignored content, including non-versioned warehouse evidence.
    names = git(
        MAIN, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ).split("\0")[:-1]
    main_commit = git(MAIN, "rev-parse", "HEAD").strip()
    for name in sorted(set(names)):
        p = MAIN / name
        if p.is_file():
            ingest(
                p,
                name,
                "original-main-worktree@" + main_commit,
                destination="archive/workspace-snapshots/original-main/" + name,
                operation="external-copy",
                zip_prefix="workspace/original-main",
            )
    # Historical runtime logs, failed mechanics and ignored checkpoints in every existing worktree.
    worktrees = [
        Path(s[9:])
        for s in git(ROOT, "worktree", "list", "--porcelain").splitlines()
        if s.startswith("worktree ")
    ]
    for tree in worktrees:
        if tree.resolve() == ROOT:
            continue
        extra = git(
            tree,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            APP + "/artifacts",
            APP + "/models",
            APP + "/logs",
            APP + "/runs",
        ).split("\0")[:-1]
        for name in extra:
            p = tree / name
            if p.is_file():
                ingest(
                    p,
                    name,
                    "local-worktree:" + tree.name,
                    destination="archive/workspace-snapshots/" + tree.name + "/" + name,
                    operation="external-copy",
                    zip_prefix="workspace/" + tree.name,
                )
        roots = [tree / APP / ".runtime"]
        for folder in roots:
            if not folder.exists():
                continue
            for parent, dirs, files in os.walk(folder):
                dirs[:] = [
                    d
                    for d in dirs
                    if d
                    not in {
                        ".git",
                        ".venv",
                        "__pycache__",
                        "matplotlib",
                        "pytest",
                        ".cache",
                    }
                    and (Path(parent) / d).resolve()
                    not in {t.resolve() for t in worktrees}
                ]
                for fn in files:
                    p = Path(parent) / fn
                    name = p.relative_to(tree).as_posix()
                    # Tracked content is already covered by the reachable Git object pass.
                    ingest(
                        p,
                        name,
                        "local-worktree:" + tree.name,
                        destination="archive/workspace-snapshots/"
                        + tree.name
                        + "/"
                        + name,
                        operation="external-copy",
                        zip_prefix="workspace/" + tree.name,
                    )
    # Every reachable historical Git file version, including deleted/intermediate source/data.
    versions = {}
    commit = None
    for line in git(
        ROOT,
        "log",
        "--all",
        "--root",
        "--format=COMMIT:%H",
        "--raw",
        "--no-abbrev",
        "--no-renames",
    ).splitlines():
        if line.startswith("COMMIT:"):
            commit = line[7:]
        elif line.startswith(":") and not line.startswith("::") and "\t" in line:
            fields, name = line.split("\t", 1)
            bits = fields.split()
            oid = bits[3]
            if oid != "0" * 40:
                versions.setdefault(
                    oid, (commit, json.loads(name) if name.startswith('"') else name)
                )
    objects = [
        s.split(" ", 1)
        for s in git(ROOT, "rev-list", "--objects", "--all").splitlines()
        if " " in s
    ]
    needed = [(oid, name) for oid, name in objects if oid not in blob_map]
    proc = subprocess.Popen(
        ["git", "-c", "safe.directory=" + str(ROOT), "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    for ix, (oid, name) in enumerate(needed):
        proc.stdin.write((oid + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().strip().split()
        if len(header) != 3:
            raise ValueError("Missing Git object " + oid)
        kind, size = header[1], int(header[2])
        dest = archive / "git-objects" / oid[:2] / oid
        if kind == "blob":
            dest.parent.mkdir(parents=True, exist_ok=True)
            f = dest.open("wb")
        else:
            f = None
        remaining = size
        while remaining:
            chunk = proc.stdout.read(min(1024 * 1024, remaining))
            remaining -= len(chunk)
            if f:
                f.write(chunk)
        proc.stdout.read(1)
        if f:
            f.close()
            version, original = versions.get(oid, ("reachable-git-history", name))
            row = ingest(
                dest,
                original,
                version,
                oid,
                destination=dest.relative_to(ROOT).as_posix(),
                operation="retain",
                zip_prefix="git-versions/" + str(version) + "-" + oid[:10],
            )
            blob_map[oid] = row
            if row["archive_path"] != dest.relative_to(ROOT).as_posix():
                dest.unlink()
        if ix % 2000 == 0:
            print("Inventory historical objects", ix, flush=True)
    proc.stdin.close()
    proc.stdout.close()
    proc.wait()
    # Map all Git origins even when their bytes already had an on-disk backing.
    for oid, (version, name) in versions.items():
        if oid in blob_map:
            blob_map[oid]["origins"].append(
                {
                    "path": name,
                    "source_version": version,
                    "git_blob": oid,
                    "representation": "git-blob",
                }
            )
    manifest["files"] = list(entries.values())
    # LFS pointers and available real payloads are distinguished explicitly.
    common = Path(git(ROOT, "rev-parse", "--git-common-dir").strip())
    common = common if common.is_absolute() else (ROOT / common).resolve()
    pending = {d.relative_to(ROOT).as_posix(): s for s, d, op in actions}
    for row in list(manifest["files"]):
        source = pending.get(row["archive_path"], ROOT / row["archive_path"])
        if row["bytes"] < 1024:
            text = source.read_bytes()
            if text.startswith(b"version https://git-lfs.github.com/spec/v1"):
                oid = next(
                    x.split(b":", 1)[1].decode()
                    for x in text.splitlines()
                    if x.startswith(b"oid sha256:")
                )
                payload = common / "lfs/objects" / oid[:2] / oid[2:4] / oid
                row["lfs_oid"] = oid
                row["lfs_missing"] = not payload.is_file()
                if payload.is_file():
                    payload_row = ingest(
                        payload,
                        "lfs-payloads/" + oid,
                        "git-lfs:" + oid,
                        destination="archive/lfs-payloads/" + oid,
                        operation="external-copy",
                        zip_prefix="payloads",
                    )
                    row["lfs_payload_path"] = payload_row["archive_path"]
    manifest["files"] = list(entries.values())
    write(archive / "manifests/pre-migration.json", manifest)
    print(
        "Pre-migration inventory sealed",
        len(entries),
        "unique file versions",
        flush=True,
    )
    # Only now move/backup legacy files, verifying resolved paths remain in this worktree.
    for source, dest, operation in actions:
        source = source.resolve()
        dest = dest.resolve()
        if not source.is_relative_to(ROOT) or not dest.is_relative_to(archive):
            raise ValueError("Unsafe migration target")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if operation == "move":
            os.replace(source, dest)
        else:
            shutil.copyfile(source, dest)
    # Retire other partial research code after its exact source is preserved.
    for p in (ROOT / APP / "src/robot_navigation").glob("*.py"):
        if p.name in KEEP:
            continue
        name = p.relative_to(ROOT).as_posix()
        dest = safe_dest("archive/historical-source/" + name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest)
        for row in manifest["files"]:
            if row["archive_path"] == name:
                row["archive_path"] = dest.relative_to(ROOT).as_posix()
                row["preservation"] = "move"
    write(archive / "manifests/pre-migration.json", manifest)
    print("Checking complete post-migration archive...", flush=True)
    integrity = verify_sources(ROOT, manifest)
    write(archive / "manifests/integrity.json", integrity)
    for collection in ("classic", "partial"):
        write(
            archive
            / "manifests"
            / (
                "legacy-classic.json"
                if collection == "classic"
                else "partial-research.json"
            ),
            {
                "files": [
                    {
                        "original_path": r["original_path"],
                        "archive_path": r["archive_path"],
                        "sha256": r["sha256"],
                        "source_version": r["source_version"],
                    }
                    for r in manifest["files"]
                    if r["collection"] in (collection, "both")
                ]
            },
        )
    fields = [
        "original_path",
        "archive_path",
        "bytes",
        "sha256",
        "git_blob",
        "category",
        "collection",
        "source_version",
        "zip_path",
    ]
    with (archive / "INDEX.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in manifest["files"])
    with (archive / "manifests/original-path-map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["original_path", "source_version", "archive_path", "sha256"])
        for r in manifest["files"]:
            for o in r["origins"]:
                w.writerow(
                    [o["path"], o["source_version"], r["archive_path"], r["sha256"]]
                )
    print(json.dumps(integrity, indent=2), flush=True)


if __name__ == "__main__":
    main()
