import argparse
import json
from pathlib import Path

import torch

from nordland_vpr import BoQVPR, predict_folder, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/user/TACVU2'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/checkpoints/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--queries-dir', type=Path, default=None)
    parser.add_argument('--database-dir', type=Path, default=None)
    parser.add_argument('--allow-demo', action='store_true')
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state.get('config', {})
    if config.get('demo_mode') and not args.allow_demo:
        raise ValueError('Refusing demo checkpoint for the official test')
    seed_everything(int(config.get('seed', 2026)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BoQVPR(config).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    result, _ = predict_folder(
        model=model,
        root=args.root / 'test',
        output_file=args.output,
        image_size=int(config.get('image_size', 320)),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        queries_dir=args.queries_dir,
        database_dir=args.database_dir,
    )
    print(json.dumps({'rows':len(result), 'output':str(args.output.resolve()),
                      'checkpoint':str(args.checkpoint.resolve()), 'device':str(device)}))


if __name__ == '__main__':
    main()
