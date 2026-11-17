"""Inspect Nordland brightness and adjacent-frame changes before cleaning train data.

Run from the project root: python scripts/frame_stats.py
Adjust the proposed masks after inspecting runs/frame_stats/brightness_and_change.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def frames(folder):
    paths = sorted(folder.glob('images-*.png'))
    if not paths:
        raise FileNotFoundError(folder)
    return {int(path.stem.rsplit('-', 1)[-1]): path for path in paths}


def frame_stats(paths):
    brightness = np.empty(len(paths), dtype=np.float32)
    changes = np.full(len(paths), np.nan, dtype=np.float32)
    previous = None
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            gray = np.asarray(image.convert('L').resize((64, 36)), dtype=np.float32) / 255
        brightness[i] = gray.mean()
        if previous is not None:
            changes[i - 1] = np.abs(previous - gray).mean()
        previous = gray
    return brightness, changes


def segments(positions, mask, min_length=1):
    result = []
    start = None
    for i in range(len(mask) + 1):
        active = i < len(mask) and bool(mask[i])
        if active and start is None:
            start = i
        if start is not None and not active:
            if i - start >= min_length:
                result.append((int(positions[start]), int(positions[i - 1])))
            start = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dark', type=float, default=0.08, help='Both seasons darker than this (default: 0.08)')
    parser.add_argument('--still', type=float, default=0.002, help='Both seasons change less than this (default: 0.002)')
    parser.add_argument('--min-stop', type=int, default=10, help='Minimum consecutive still frames (default: 10)')
    args = parser.parse_args()
    if not 0 < args.dark < 1 or not 0 < args.still < 1 or args.min_stop < 2:
        parser.error('Expected 0 < dark, still < 1 and min-stop >= 2')

    root = Path(__file__).resolve().parents[1]
    data = root / 'data' / 'nordland-part-2020'
    out = root / 'runs' / 'frame_stats'
    out.mkdir(parents=True, exist_ok=True)
    summer, winter = frames(data / 'summer'), frames(data / 'winter')
    if summer.keys() != winter.keys():
        raise ValueError('Summer and winter positions differ')
    positions = np.array(sorted(summer))
    if not np.all(np.diff(positions) == 1):
        raise ValueError('Frame positions must be consecutive to measure adjacent-frame change')
    sb, sd = frame_stats([summer[int(p)] for p in positions])
    wb, wd = frame_stats([winter[int(p)] for p in positions])
    dark = (sb < args.dark) & (wb < args.dark)
    # diff at i compares frame i with i+1; both endpoints must be visible.
    still_edge = (sd < args.still) & (wd < args.still) & ~dark
    still_edge[:-1] &= ~dark[1:]
    stop_ranges = [(start, end + 1) for start, end in segments(positions, still_edge, args.min_stop - 1)]
    stop = np.zeros(len(positions), dtype=bool)
    for start, end in stop_ranges:
        stop[(positions >= start) & (positions <= end)] = True
    table = pd.DataFrame({'position': positions, 'split': np.where(positions <= 1640, 'train',
                         np.where(positions >= 1651, 'validation', 'buffer')),
                         'summer_brightness': sb, 'winter_brightness': wb,
                         'summer_next_diff': sd, 'winter_next_diff': wd,
                         'tunnel_candidate': dark, 'stop_candidate': stop})
    table.to_csv(out / 'frame_stats.csv', index=False)
    ranges = [('tunnel', a, b) for a, b in segments(positions, dark)]
    ranges += [('stop', a, b) for a, b in stop_ranges]
    summary = pd.DataFrame(ranges, columns=['type', 'start', 'end'])
    if not summary.empty:
        summary['frames'] = summary['end'] - summary['start'] + 1
        summary['split'] = np.where(summary['end'] <= 1640, 'train',
                            np.where(summary['start'] >= 1651, 'validation', 'boundary'))
        summary = summary.sort_values(['start', 'type'])
    summary.to_csv(out / 'candidate_segments.csv', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, summer_values, winter_values in [(axes[0], sb, wb), (axes[1], sd, wd)]:
        ax.plot(positions, summer_values, label='summer', linewidth=0.8)
        ax.plot(positions, winter_values, label='winter', linewidth=0.8)
        ax.axvspan(1640.5, 1650.5, color='gray', alpha=0.15, label='split buffer')
        for start, end in segments(positions, dark):
            ax.axvspan(start, end, color='black', alpha=0.12)
        for start, end in stop_ranges:
            ax.axvspan(start, end, color='orange', alpha=0.15)
        ax.legend(loc='upper right')
    axes[0].axhline(args.dark, color='black', linestyle=':', linewidth=1)
    axes[1].axhline(args.still, color='orange', linestyle=':', linewidth=1)
    axes[0].set_ylabel('Mean grayscale brightness')
    axes[1].set_ylabel('Mean absolute change to next frame')
    axes[1].set_xlabel('Position (black shade: tunnel candidate; orange: stop candidate)')
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / 'brightness_and_change.png', dpi=150)
    plt.close(fig)

    print(f'Positions: {len(table)} | train: {(table.split == "train").sum()} | val: {(table.split == "validation").sum()}')
    print(f'Tunnel candidates: {int(dark.sum())} frames | stop candidates: {int(stop.sum())} frames')
    print(summary.to_string(index=False) if not summary.empty else 'No candidate segments')
    print('Saved:', out)


if __name__ == '__main__':
    main()
