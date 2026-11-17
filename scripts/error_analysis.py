"""Phân tích lỗi trên tập validation, dùng file dự đoán đã lưu của các run.

Cách dùng trong notebook (đặt sau cell định nghĩa WORK_ROOT và DATA_ROOT):
    %run scripts/error_analysis.py --center 1770
Hoặc chạy từ PowerShell: python scripts/error_analysis.py --center 1770

Đầu vào: runs/*/predictions/validation_predictions.csv (cột query_id, winter_id),
được lưu từ checkpoint best của mỗi run. Position lấy từ số trong tên ảnh, ví dụ
images-01651 -> 1651.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

TOL = 10            # dung sai của đề
WINDOW = 20         # cửa sổ trượt khi vẽ độ chính xác theo position
parser = argparse.ArgumentParser(description='Analyze validation errors and inspect a route segment.')
parser.add_argument('--center', type=int, help='Position to inspect; default is the hardest segment')
args = parser.parse_args()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / 'runs'
VAL_ROOT = PROJECT_ROOT / 'data' / 'nordland-part-2020'
OUT = RUNS_ROOT / 'error_analysis'
OUT.mkdir(exist_ok=True)


def pos(image_id):
    return int(re.search(r'(\d+)', str(image_id)).group(1))


# 1. Gom dự đoán của mọi run thành một bảng: hàng = run, cột = query --------
frames = {}
for path in sorted(RUNS_ROOT.glob('*/predictions/validation_predictions.csv')):
    df = pd.read_csv(path, dtype=str)
    df = df.drop_duplicates('query_id')          # đề chỉ chấm dòng đầu tiên
    q = df.query_id.map(pos)
    w = df.winter_id.map(pos)
    frames[path.parent.parent.name] = pd.Series((w - q).to_numpy(), index=q.to_numpy())

err = pd.DataFrame(frames).T.sort_index(axis=1)  # run x query_position, sai số có dấu
correct = err.abs() <= TOL
print(f'{len(err)} run, {err.shape[1]} query')
print((correct.mean(axis=1) * 100).round(1).sort_values(ascending=False).to_string())

# 2. Query đúng ở mọi run / sai ở mọi run / lúc đúng lúc sai ------------------
n_ok = correct.sum(axis=0)
always_ok = (n_ok == len(err)).mean() * 100
always_bad = (n_ok == 0).mean() * 100
mixed = 100 - always_ok - always_bad
oracle = (n_ok > 0).mean() * 100
print(f'\nĐúng ở mọi run: {always_ok:.1f}% | Sai ở mọi run: {always_bad:.1f}% | '
      f'Lúc đúng lúc sai: {mixed:.1f}%')
print(f'Trần nếu luôn chọn được run đúng (oracle): {oracle:.1f}%')

# 3. Dải đúng/sai theo position + độ chính xác trượt --------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 2 + 0.35 * len(err) + 3), sharex=True,
                         gridspec_kw={'height_ratios': [0.35 * len(err) + 1, 3]})
positions = err.columns.to_numpy()
axes[0].imshow(correct.to_numpy(), aspect='auto', cmap='RdYlGn', interpolation='nearest',
               extent=[positions[0] - 0.5, positions[-1] + 0.5, len(err) - 0.5, -0.5])
axes[0].set_yticks(range(len(err)))
axes[0].set_yticklabels([name.replace('boq_nordland_40epochs_', '') for name in err.index],
                        fontsize=7)
axes[0].set_title('Đúng (xanh) / sai (đỏ) theo position của query')
rolling = correct.T.rolling(WINDOW, center=True, min_periods=1).mean() * 100
axes[1].plot(rolling.index, rolling.mean(axis=1), color='black', lw=2, label='trung bình các run')
axes[1].fill_between(rolling.index, rolling.min(axis=1), rolling.max(axis=1),
                     color='gray', alpha=0.3, label='khoảng min–max giữa các run')
axes[1].set_ylabel(f'R@1 trượt ({WINDOW} frame), %')
axes[1].set_xlabel('Position của query')
axes[1].legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'correct_by_position.png', dpi=130)
plt.show()

# 4. Kiểu lỗi: sai gần hay sai xa, lệch về phía trước hay phía sau ----------------
wrong = err.where(~correct)
bins = [10, 20, 50, 100, 1000]
labels = ['11–20', '21–50', '51–100', '>100']
dist = wrong.abs().stack().pipe(pd.cut, bins=bins, labels=labels).value_counts(normalize=True)
print('\nPhân bố độ lệch của các dự đoán sai (gộp mọi run):')
print((dist.reindex(labels) * 100).round(1).to_string())
ahead = (wrong.stack() > 0).mean() * 100
print(f'Dự đoán sai nằm phía trước vị trí thật: {ahead:.1f}% (khoảng 50% là không lệch hướng)')

# 5. Hubness: ảnh database nào bị chọn quá nhiều lần ---------------------------
hub_rows = []
for run, series in err.iterrows():
    winter = series.index.to_numpy() + series.to_numpy()      # position ảnh đông được chọn
    ok = np.abs(series.to_numpy()) <= TOL
    counts = pd.Series(winter).value_counts()
    wrong_counts = pd.Series(winter[~ok]).value_counts()
    top10 = counts.head(10)
    hub_rows.append({
        'run': run.replace('boq_nordland_40epochs_', ''),
        'distinct_db_chosen': counts.size,
        'top10_share_%': round(top10.sum() / len(winter) * 100, 1),
        'max_count': int(counts.iloc[0]),
        'wrong_into_top5_hubs_%': round(wrong_counts.head(5).sum() / max((~ok).sum(), 1) * 100, 1),
    })
hubs = pd.DataFrame(hub_rows)
print('\nHubness (350 ảnh database, 350 query):')
print(hubs.to_string(index=False))

best_run = correct.mean(axis=1).idxmax()
series = err.loc[best_run]
winter = series.index.to_numpy() + series.to_numpy()
ok = np.abs(series.to_numpy()) <= TOL
fig, ax = plt.subplots(figsize=(14, 3))
all_counts = pd.Series(winter).value_counts().sort_index()
bad_counts = pd.Series(winter[~ok]).value_counts().reindex(all_counts.index, fill_value=0)
ax.bar(all_counts.index, all_counts.values, width=1, color='lightgray', label='tổng số lần được chọn')
ax.bar(bad_counts.index, bad_counts.values, width=1, color='crimson', label='trong đó là dự đoán sai')
ax.set_xlabel('Position của ảnh database')
ax.set_ylabel('Số query chọn ảnh này')
ax.set_title(f'Số lần mỗi ảnh database được chọn làm top-1 ({best_run})', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'hub_counts.png', dpi=130)
plt.show()

# 6. Xem tận mắt đoạn khó nhất ------------------------------------------------
hard = (~correct).mean(axis=0).rolling(WINDOW, center=True, min_periods=1).mean()
center = args.center if args.center is not None else int(hard.idxmax())
print(f'\nQuan sát đoạn quanh position {center}; đoạn khó nhất tự động quanh {int(hard.idxmax())}')


def find_image(season, position):
    hits = list(VAL_ROOT.rglob(f'{season}/*{position:05d}*'))
    return Image.open(hits[0]).convert('RGB') if hits else None


sample = [p for p in range(center - WINDOW // 2, center + WINDOW // 2 + 1, 4) if p in err.columns][:5]
if not sample:
    raise ValueError(f'No validation queries near position {center}')
fig, axes = plt.subplots(3, len(sample), figsize=(3.2 * len(sample), 6))
for col, p in enumerate(sample):
    predicted = p + int(series[p])
    for row, (season, position, title) in enumerate([
        ('summer', p, f'Query hè {p}'),
        ('winter', p, f'Đông đúng {p}'),
        ('winter', predicted, f'Đông dự đoán {predicted}'),
    ]):
        image = find_image(season, position)
        ax = axes[row, col]
        if image is not None:
            ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
fig.suptitle(f'Đoạn khó nhất, dự đoán theo {best_run}', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / (f'segment_{center}.png' if args.center is not None else 'hardest_segment.png'), dpi=130)
plt.show()

err.to_csv(OUT / 'signed_errors_by_run.csv')
hubs.to_csv(OUT / 'hubness.csv', index=False)
print('\nĐã lưu vào', OUT)
