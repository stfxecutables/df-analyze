from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import re
import sys
from argparse import ArgumentParser
from pathlib import Path


def load_log_text() -> tuple[str, Path]:
    parser = ArgumentParser()
    parser.add_argument("logfile", type=Path)
    args = parser.parse_args()
    logfile = Path(args.logfile)
    outfile = logfile.parent / f"{logfile.stem}.clean{logfile.suffix}"
    text = logfile.read_text()
    return text, outfile


def remove_warnings(lines: list[str]) -> list[str]:
    regs = [
        "UserWarning: The verbose",
        "is deprecated",
        "FutureWarning",
        r"warnings.warn\(",
    ]
    precleaned = []
    for line in lines:
        matches = [re.search(reg, line) is not None for reg in regs]
        if any(matches):
            continue
        precleaned.append(line)
    return precleaned


def clean_text(text: str) -> str:
    regs = [
        r"\|[ ▏▎▍▌▋▊▉█▏████████████████]+\|",
        "Best trial",
        "__UNSORTED",  # hack for loading bug
    ]
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]

    precleaned = remove_warnings(lines)

    cleaned = []
    for i, line in enumerate(precleaned[:-1]):
        is_header = re.search("Tuning .* for selection=", line) is not None
        nxt_header = re.search("Tuning .* for selection=", precleaned[i + 1]) is not None
        if is_header or nxt_header:
            cleaned.append(line)
            continue

        matches = [re.search(reg, line) is not None for reg in regs]
        if any(matches):
            continue
        cleaned.append(line)
    cleaned.append(lines[-1])
    cleaned = [line for line in cleaned if line != ""]
    clean = "\n".join(cleaned)
    return clean


if __name__ == "__main__":
    text, outfile = load_log_text()
    clean = clean_text(text)
    outfile.write_text(clean)
    print(clean)
