# Nordland Cross-Season VPR — Bag-of-Queries

This implementation follows `docs/plan.md` in the T-Sunm/Nordland-Cross-Season-VPR repository and the BoQ architecture from *A Place is Worth a Bag of Learnable Queries* (CVPR 2024).

## Task

Query images show the Nordland railway in summer; database images show the same route in winter. For each summer image the model returns exactly one winter image ID. A prediction is correct when the absolute frame-position difference is at most 10. The leaderboard metric is Recall@1 under this position tolerance.

## Modes and data use

`notebooks/BoQ_Nordland_OLPAI2026.ipynb` has two distinct modes, controlled by `DATA_MODE` in the configuration cell.

- `tutorial`: expects `data/nordland-part-2020.zip`, with aligned summer/winter images numbered 251–2000. The train split uses 251–1640, frames 1641–1650 are excluded, and validation uses 1651–2000. The notebook generates a JPEG mock test with randomized identifiers.
- `official`: reads only the competition's `train`, `validation` and labeled CSV files under `/home/user/TACVU2`. Its final inference uses `test/queries` and `test/database`, and does not read hidden test labels.

**Competition compliance:** The OLP'AI 2026 rules prohibit outside datasets (including the standalone Nordland tutorial data) and pretrained weights not supplied by the organizers. Use `DATA_MODE='official'` and only organizer-authorized local weights for an official submission. The notebook and scripts never download weights. With `APPROVED_RESNET_WEIGHTS=None`, ResNet-50 starts from random initialization and the GAP baseline is not an ImageNet pretrained baseline.

## Setup

Requires Python 3.10+, PyTorch, torchvision, numpy, pandas, Pillow, matplotlib, Jupyter and pytorch-metric-learning.

```bash
pip install torch torchvision numpy pandas pillow matplotlib jupyter pytorch-metric-learning
```

Start Jupyter in the repository root so the `data/` and `artifacts/` paths resolve correctly.

```bash
jupyter notebook notebooks/BoQ_Nordland_OLPAI2026.ipynb
```

The first run uses `DEMO_MODE=True`, which limits training to one epoch and fewer places. Demo metrics are only smoke-test results. Set `DEMO_MODE=False` for full experiments. Supply the authorized ResNet-50 weights through `APPROVED_RESNET_WEIGHTS` when available.

## Training and evaluation

The notebook creates one summer and one winter view per training position, then batches multiple positions for metric learning. The backbone is ResNet-50 truncated after layer3 (1024-channel feature maps). A 3×3 projection feeds local tokens into BoQ blocks. Each block applies a Transformer encoder, self-attention over learned queries, and cross-attention to the local tokens. Outputs are projected into an L2-normalized global descriptor.

The training objective is Multi-Similarity Loss (`alpha=1`, `beta=50`, `base=0`) with the Multi-Similarity Miner (`epsilon=0.1`), using AdamW and an epoch-based warmup/multistep learning-rate schedule. The best checkpoint is selected by validation Accuracy@±10; Recall@5, Recall@10, retrieval errors and qualitative examples are also recorded.

The `artifacts/` directory is created automatically, with `checkpoints/`, `embeddings/`, `figures/`, `mock_test/`, evaluation CSVs and `output.csv`.

## Official inference

After training a **non-demo model on permitted data**, place its checkpoint in `artifacts/checkpoints/best.pt`, then run:

```bash
python main.py --root /home/user/TACVU2 --checkpoint artifacts/checkpoints/best.pt --output submission.csv
```

The script reads only `test/queries/*.jpg` and `test/database/*.jpg`, computes descriptors, performs cosine similarity retrieval, verifies that every query has exactly one valid database prediction and writes `submission.csv` with exactly `query_id,winter_id` columns. It rejects demo checkpoints unless `--allow-demo` is passed explicitly. To run on the unlocked private queries, pass `--queries-dir /home/user/TACVU2/private_test/queries` while keeping the same official `test/database`.

The competition also asks for a ZIP containing `output.csv` and the generating notebook. Copy the appropriate prediction CSV to `output.csv` when preparing that submission; keep the `Final/Tac_vu_2/main.py`, checkpoint, and supporting module for reproducibility.

The 20-minute total runtime limit for the final inference must be measured on the contest machine; it cannot be guaranteed without the actual GPU, dataset and checkpoint.

## Files

```text
notebooks/BoQ_Nordland_OLPAI2026.ipynb
nordland_vpr.py
main.py
requirements.txt
tests/test_pipeline.py
```

## References

- Ali-bey, A., Chaib-draa, B., & Giguère, P. (2024). *BoQ: A Place is Worth a Bag of Learnable Queries*. CVPR.
- T-Sunm/Nordland-Cross-Season-VPR, `docs/plan.md`.
- OLP'AI 2026, Task 2: Định vị tàu hoả bốn mùa.
