from pathlib import Path
import csv
from collections import Counter


OUT = Path(__file__).resolve().parents[1] / "data" / "splits" / "nordland_clean_4fold.csv"
MOVING = range(251, 1287)
BUFFER = 30


def main():
    positions = list(MOVING)
    block_size = len(positions) // 4
    rows = []

    for fold in range(4):
        start = fold * block_size
        stop = (fold + 1) * block_size
        val = set(positions[start:stop])
        lo, hi = min(val), max(val)

        for position in range(251, 2001):
            if position in val:
                split = "validation"
            elif position in MOVING:
                split = "buffer" if lo - BUFFER <= position <= hi + BUFFER else "train"
            elif 1796 <= position <= 1871:
                split = "holdout"
            elif position >= 1872:
                split = "tunnel_excluded"
            else:
                split = "stopped_excluded"
            rows.append((fold, position, split))

    for fold in range(4):
        part = [(position, split) for f, position, split in rows if f == fold]
        train = [position for position, split in part if split == "train"]
        val = [position for position, split in part if split == "validation"]
        assert len(val) == 259 and sum(split == "holdout" for _, split in part) == 76
        assert min(abs(a - b) for a in train for b in val) > BUFFER

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "position", "split"])
        writer.writerows(rows)
    for (fold, split), count in sorted(Counter((f, s) for f, _, s in rows).items()):
        print(f"{fold} {split:17} {count}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
