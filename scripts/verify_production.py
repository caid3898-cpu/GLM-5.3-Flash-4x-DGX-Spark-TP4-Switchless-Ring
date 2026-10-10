"""Read-only production tree and externally pinned profile verification."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, profile, profile_sha256, closure_sha256):
    root = Path(root).resolve()
    profile = (root / profile).resolve()
    if (not profile.is_relative_to(root) or profile.name != "candidate.env" or
            sha(profile) != profile_sha256):
        raise ValueError("selected production profile mismatch")
    closure = root / "SOURCE_CLOSURE.json"
    if sha(closure) != closure_sha256:
        raise ValueError("externally pinned production closure mismatch")
    data = json.loads(closure.read_text())
    files = data.get("files")
    if (data.get("schema") != "night-production-source-v1" or
            data.get("status") != "PREPARED_SOURCE_ONLY" or
            type(files) is not dict or not files):
        raise ValueError("production closure schema")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("production tree symlink refused")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"SOURCE_CLOSURE.json"}:
        raise ValueError("production file inventory drift")
    for name, digest in files.items():
        rel = Path(name)
        path = root / rel
        if (rel.is_absolute() or ".." in rel.parts or path.is_symlink() or
                not path.is_file() or sha(path) != digest):
            raise ValueError("production file changed: " + name)
    if files.get("candidate.env") != profile_sha256:
        raise ValueError("profile not in production closure")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--profile-sha256", required=True)
    p.add_argument("--closure-sha256", required=True)
    a = p.parse_args()
    result = verify(a.root, a.profile, a.profile_sha256, a.closure_sha256)
    print(json.dumps({"status":"PASS","option":result["option"],"files":len(result["files"])}))


if __name__ == "__main__":
    main()
