"""Select Chapter 11 evaluation without skipping the required workflow gate."""
import os
from pathlib import Path
import subprocess


def relevant(paths: list[bytes]) -> bool:
    return any(path.startswith((b"Chapter 11/", b".github/workflows/")) for path in paths)


def main() -> None:
    base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
    # A new branch has no previous commit: evaluate it conservatively.
    if not base or set(base) == {"0"}:
        run = True
    else:
        result = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, head, "--"],
            check=True, capture_output=True,
        )
        run = relevant(result.stdout.split(b"\0"))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"relevant={str(run).lower()}\n")


if __name__ == "__main__":
    main()
