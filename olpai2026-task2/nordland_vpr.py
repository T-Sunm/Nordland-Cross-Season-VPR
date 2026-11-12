import csv
import json
import math
import os
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms as T
from torchvision.transforms import InterpolationMode


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_nordland(zip_path, output_dir):
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = Path(item.filename)
            if relative.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            destination = (root / relative).resolve()
            if root not in destination.parents:
                raise ValueError(f"Unsafe archive entry: {item.filename}")
            if destination.is_file() and destination.stat().st_size == item.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open('wb') as target:
                shutil.copyfileobj(source, target)
    return root


def position_from_name(path):
    match = re.search(r'(\d+)$', Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse position: {path}")
    return int(match.group(1))


def tutorial_metadata(directory, require_complete=True):
    root = Path(directory)
    records = []
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        seasons = [name.lower() for name in path.parts if name.lower() in {'summer', 'winter'}]
        if len(seasons) != 1:
            continue
        position = position_from_name(path)
        split = 'train' if position <= 1640 else 'buffer' if position <= 1650 else 'validation'
        records.append({'path': str(path.resolve()), 'id': path.stem,
                        'season': seasons[0], 'position': position, 'split': split})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FileNotFoundError(f"No Nordland summer/winter images in {root}")
    if frame.duplicated(['season', 'position']).any():
        raise ValueError('Duplicate positions found in one season')
    observed = {s: set(frame.loc[frame.season == s, 'position']) for s in ['summer', 'winter']}
    if observed['summer'] != observed['winter']:
        raise ValueError('Summer and winter position ranges differ')
    if require_complete:
        expected = set(range(251, 2001))
        if observed['summer'] != expected:
            missing = sorted(expected - observed['summer'])
            extra = sorted(observed['summer'] - expected)
            raise ValueError(f"Unexpected positions: missing={missing[:20]}, extra={extra[:20]}")
    return frame.sort_values(['position', 'season']).reset_index(drop=True)


def official_metadata(data_root):
    root = Path(data_root)
    parts = []
    for split in ['train', 'validation']:
        for season in ['summer', 'winter']:
            label_file = root / f'{split}_{season}.csv'
            if not label_file.is_file():
                raise FileNotFoundError(label_file)
            df = pd.read_csv(label_file)
            id_col = f'{season}_id'
            if id_col not in df or 'position' not in df:
                raise ValueError(f"Expected {id_col},position in {label_file}")
            images = root / split / season
            for item in df.itertuples(index=False):
                image_id = str(getattr(item, id_col))
                path = images / f'{image_id}.jpg'
                if not path.is_file():
                    raise FileNotFoundError(path)
                parts.append({'path': str(path.resolve()), 'id': image_id,
                              'season': season, 'position': int(item.position), 'split': split})
    return pd.DataFrame(parts)


def make_training_pairs(metadata, split='train', max_positions=None, seed=2026, tolerance=10):
    df = metadata.loc[metadata.split == split]
    summer = df.loc[df.season == 'summer'].sort_values('position')
    winter = df.loc[df.season == 'winter'].sort_values('position')
    if summer.empty or winter.empty:
        raise ValueError(f"Missing summer or winter images in {split}")
    winter_pos = winter.position.to_numpy(dtype=np.int64)
    records = []
    for item in summer.itertuples(index=False):
        index = int(np.searchsorted(winter_pos, item.position))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(winter_pos)]
        best = min(candidates, key=lambda i: abs(winter_pos[i] - item.position))
        if abs(int(winter_pos[best]) - item.position) > tolerance:
            continue
        target = winter.iloc[best]
        label = int(item.position)
        records.extend([
            {'path': item.path, 'position': label, 'season': 'summer'},
            {'path': target.path, 'position': label, 'season': 'winter'}
        ])
    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise ValueError('No valid cross-season training pairs')
    if max_positions is not None:
        positions = pairs.position.unique()
        rng = np.random.default_rng(seed)
        selected = set(rng.choice(positions, min(max_positions, len(positions)), replace=False).tolist())
        pairs = pairs.loc[pairs.position.isin(selected)]
    return pairs.reset_index(drop=True)


def image_transform(size=320, train=False):
    operations = [T.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)]
    if train:
        operations += [T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
                       T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.3))], p=0.15)]
    operations += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(operations)


class ImageTable(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        item = self.table.iloc[index]
        with Image.open(item.path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(item.position), str(item.season), str(item.path)


class PairBatchSampler(Sampler):
    def __init__(self, pair_table, places_per_batch=16, seed=2026, drop_last=False):
        self.positions = {}
        for index, item in pair_table.reset_index(drop=True).iterrows():
            self.positions.setdefault(int(item.position), []).append(index)
        if any(len(indices) != 2 for indices in self.positions.values()):
            raise ValueError('Each position must contain one summer and one winter image')
        self.labels = sorted(self.positions)
        self.places_per_batch = int(places_per_batch)
        if self.places_per_batch < 2:
            raise ValueError('places_per_batch must be at least 2')
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = drop_last

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        labels = rng.permutation(self.labels).tolist()
        for start in range(0, len(labels), self.places_per_batch):
            group = labels[start:start + self.places_per_batch]
            if self.drop_last and len(group) < self.places_per_batch:
                continue
            indices = [idx for label in group for idx in self.positions[label]]
            yield rng.permutation(indices).tolist()

    def __len__(self):
        count = len(self.labels)
        return count // self.places_per_batch if self.drop_last else math.ceil(count / self.places_per_batch)


class ResNetBackbone(nn.Module):
    def __init__(self, local_weights=None, unfreeze_layer3=True):
        super().__init__()
        resnet = models.resnet50(weights=None)
        if local_weights:
            state = torch.load(local_weights, map_location='cpu', weights_only=True)
            for key in ('state_dict', 'model_state_dict', 'model'):
                if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            result = resnet.load_state_dict(state, strict=False)
            if len(result.missing_keys) > 2 or result.unexpected_keys:
                raise ValueError('Weights are not compatible with torchvision ResNet-50')
        self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.out_channels = 1024
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        if unfreeze_layer3:
            for parameter in self.features[-1].parameters():
                parameter.requires_grad = True

    def forward(self, images):
        return self.features(images)

    def train(self, mode=True):
        super().train(mode)
        for block in self.features[:-1]:
            block.eval()
        if not any(p.requires_grad for p in self.features[-1].parameters()):
            self.features[-1].eval()
        return self


class GAPModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        features = self.backbone(images)
        return F.normalize(F.adaptive_avg_pool2d(features, 1).flatten(1), dim=1)


class BoQBlock(nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super().__init__()
        if in_dim % nheads:
            raise ValueError('in_dim must be divisible by nheads')
        self.encoder = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads,
                                                  dim_feedforward=4 * in_dim,
                                                  dropout=0.0, batch_first=True)
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = nn.MultiheadAttention(in_dim, nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)
        self.cross_attn = nn.MultiheadAttention(in_dim, nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)

    def forward(self, x, return_attention=False):
        x = self.encoder(x)
        queries = self.queries.expand(x.size(0), -1, -1)
        queries = self.norm_q(queries + self.self_attn(queries, queries, queries, need_weights=False)[0])
        output, attention = self.cross_attn(queries, x, x, need_weights=return_attention)
        return x, self.norm_out(output), attention


class BoQ(nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32,
                 num_layers=2, output_dim=8192, nheads=8):
        super().__init__()
        if proj_channels % nheads or output_dim % proj_channels:
            raise ValueError('Invalid projection, attention head or output dimensions')
        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        self.boqs = nn.ModuleList([BoQBlock(proj_channels, num_queries, nheads)
                                   for _ in range(num_layers)])
        self.fc = nn.Linear(num_layers * num_queries, output_dim // proj_channels)
        self.output_dim = output_dim

    def forward(self, features, return_attention=False):
        x = self.proj_c(features).flatten(2).transpose(1, 2)
        x = self.norm_input(x)
        outputs, attentions = [], []
        for block in self.boqs:
            x, output, attention = block(x, return_attention)
            outputs.append(output)
            if return_attention:
                attentions.append(attention)
        merged = torch.cat(outputs, dim=1)
        descriptor = self.fc(merged.transpose(1, 2)).flatten(1)
        descriptor = F.normalize(descriptor, dim=1)
        return (descriptor, attentions) if return_attention else descriptor


class BoQVPR(nn.Module):
    def __init__(self, config=None, local_weights=None):
        super().__init__()
        self.config = dict(config or {})
        self.backbone = ResNetBackbone(local_weights=local_weights,
                                      unfreeze_layer3=self.config.get('unfreeze_layer3', True))
        self.aggregator = BoQ(in_channels=1024,
                              proj_channels=self.config.get('proj_channels', 512),
                              num_queries=self.config.get('num_queries', 32),
                              num_layers=self.config.get('num_layers', 2),
                              output_dim=self.config.get('output_dim', 8192),
                              nheads=self.config.get('nheads', 8))

    def forward(self, images, return_attention=False):
        return self.aggregator(self.backbone(images), return_attention)


@torch.inference_mode()
def encode_images(model, table, transform, device, batch_size=32, num_workers=0):
    if len(table) == 0:
        raise ValueError('Cannot embed an empty image table')
    loader = DataLoader(ImageTable(table, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    model.eval()
    output, positions, paths = [], [], []
    for images, label, _, path in loader:
        embeddings = model(images.to(device, non_blocking=True))
        output.append(embeddings.detach().float().cpu().numpy())
        positions.extend(label.tolist())
        paths.extend(path)
    return np.concatenate(output), np.asarray(positions, dtype=np.int64), paths


def topk_search(queries, database, k=10, chunk_size=128):
    queries = np.asarray(queries, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)
    if queries.ndim != 2 or database.ndim != 2 or queries.shape[1] != database.shape[1]:
        raise ValueError('Descriptor shapes are incompatible')
    if not len(database):
        raise ValueError('Database is empty')
    k = min(int(k), len(database))
    indices, scores = [], []
    for start in range(0, len(queries), chunk_size):
        similarities = queries[start:start + chunk_size] @ database.T
        candidate = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(similarities, candidate, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        indices.append(np.take_along_axis(candidate, order, axis=1))
        scores.append(np.take_along_axis(candidate_scores, order, axis=1))
    return np.concatenate(indices, axis=0), np.concatenate(scores, axis=0)


def retrieval_metrics(query_positions, database_positions, indices, threshold=10):
    errors = np.abs(database_positions[indices] - query_positions[:, None])
    metrics = {'accuracy_pm10': float(np.mean(errors[:, 0] <= threshold) * 100),
               'median_top1_error': float(np.median(errors[:, 0])),
               'mean_top1_error': float(np.mean(errors[:, 0]))}
    for k in (5, 10):
        if indices.shape[1] >= k:
            metrics[f'recall_at_{k}'] = float(np.mean(np.any(errors[:, :k] <= threshold, axis=1)) * 100)
    return metrics, errors


def evaluate_model(model, metadata, transform, device, batch_size=32, workers=0, max_queries=None):
    validation = metadata.loc[metadata.split == 'validation']
    queries = validation.loc[validation.season == 'summer'].sort_values('position')
    database = validation.loc[validation.season == 'winter'].sort_values('position')
    if max_queries is not None:
        queries = queries.iloc[:max_queries]
        database = database.iloc[:max(10, max_queries * 10)]
    q_vec, q_pos, q_paths = encode_images(model, queries, transform, device, batch_size, workers)
    d_vec, d_pos, d_paths = encode_images(model, database, transform, device, batch_size, workers)
    indices, scores = topk_search(q_vec, d_vec, k=10)
    metrics, errors = retrieval_metrics(q_pos, d_pos, indices)
    return metrics, {'queries': queries.reset_index(drop=True), 'database': database.reset_index(drop=True),
                     'q_vec': q_vec, 'd_vec': d_vec, 'indices': indices,
                     'scores': scores, 'errors': errors}


def make_optimizer(model, lr=1e-4, weight_decay=1e-4):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, epochs, warmup_epochs=2, milestones=(8, 16), gamma=0.1):
    warmup_epochs = min(max(0, int(warmup_epochs)), max(0, epochs - 1))
    decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if warmup_epochs == 0:
        return decay
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                end_factor=1.0, total_iters=warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay],
                                                  milestones=[warmup_epochs])


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, config, scaler=None):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'epoch': epoch,
               'best_score': best_score, 'config': dict(config)}
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    if scaler is not None and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    return state


def train_model(model, train_table, validation, config, artifacts, device, resume=None):
    from pytorch_metric_learning import losses, miners

    train_set = ImageTable(train_table, image_transform(config['image_size'], train=True))
    sampler = PairBatchSampler(train_table, config['places_per_batch'], seed=config['seed'])
    loader = DataLoader(train_set, batch_sampler=sampler,
                        num_workers=config.get('workers', 0), pin_memory=device.type == 'cuda')
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50.0, base=0.0)
    optimizer = make_optimizer(model, config['lr'], config['weight_decay'])
    scheduler = make_scheduler(optimizer, config['epochs'], config['warmup_epochs'],
                               config['milestones'], config['lr_gamma'])
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    checkpoints = Path(artifacts) / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    first_epoch, best_score = 0, -1.0
    if resume:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, scaler)
        first_epoch, best_score = int(saved['epoch']) + 1, float(saved['best_score'])
    history = []
    for epoch in range(first_epoch, config['epochs']):
        sampler.epoch = epoch
        model.train()
        total_loss, steps = 0.0, 0
        for images, labels, _, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                descriptors = model(images)
                pairs = miner(descriptors.float(), labels)
                loss = criterion(descriptors.float(), labels, pairs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            steps += 1
            if config.get('demo_mode') and steps >= 2:
                break
        lr = float(optimizer.param_groups[0]['lr'])
        metrics, _ = evaluate_model(model, validation, image_transform(config['image_size']),
                                    device, config['eval_batch_size'], config.get('workers', 0),
                                    max_queries=20 if config.get('demo_mode') else None)
        score = metrics['accuracy_pm10']
        improved = score > best_score
        if improved:
            best_score = score
        scheduler.step()
        if improved:
            save_checkpoint(checkpoints / 'best.pt', model, optimizer, scheduler,
                            epoch, best_score, config, scaler)
        save_checkpoint(checkpoints / 'last.pt', model, optimizer, scheduler,
                        epoch, best_score, config, scaler)
        entry = {'epoch': epoch + 1, 'loss': total_loss / max(steps, 1), 'lr': lr,
                 **metrics, 'best_accuracy_pm10': best_score}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    pd.DataFrame(history).to_csv(Path(artifacts) / 'training_history.csv', index=False)
    return history, checkpoints / 'best.pt'


def make_mock_test(metadata, artifacts, seed=2026, query_stride=10):
    artifacts = Path(artifacts)
    root = artifacts / 'mock_test'
    if root.exists():
        shutil.rmtree(root)
    (root / 'database').mkdir(parents=True)
    (root / 'queries').mkdir(parents=True)
    rng = random.Random(seed)
    validation = metadata.loc[metadata.split == 'validation']
    winter = validation.loc[validation.season == 'winter'].sort_values('position')
    summer = validation.loc[validation.season == 'summer'].sort_values('position').iloc[::query_stride]
    records = []
    for season, table, folder, prefix in [('winter', winter, 'database', 'w'),
                                          ('summer', summer, 'queries', 'q')]:
        for item in table.itertuples(index=False):
            image_id = f'{prefix}_{rng.getrandbits(64):016x}'
            destination = root / folder / f'{image_id}.jpg'
            with Image.open(item.path) as image:
                image.convert('RGB').save(destination, quality=90)
            records.append({'id': image_id, 'season': season, 'position': int(item.position)})
    pd.DataFrame(records).to_csv(artifacts / 'mock_ground_truth.csv', index=False)
    return root


def folder_table(folder, season):
    paths = sorted(path for path in Path(folder).glob('*.jpg') if path.is_file())
    if not paths:
        raise FileNotFoundError(f'No .jpg images in {folder}')
    return pd.DataFrame([{'path': str(path.resolve()), 'id': path.stem,
                          'season': season, 'position': -1} for path in paths])


def validate_submission(submission, query_ids, database_ids):
    if list(submission.columns) != ['query_id', 'winter_id']:
        raise ValueError('Expected exactly query_id,winter_id columns')
    q = list(map(str, query_ids))
    d = set(map(str, database_ids))
    if len(submission) != len(q):
        raise ValueError(f'Expected {len(q)} predictions, got {len(submission)}')
    if submission.query_id.isna().any() or submission.winter_id.isna().any():
        raise ValueError('Submission contains empty IDs')
    if submission.query_id.duplicated().any():
        raise ValueError('Duplicate query_id')
    if set(submission.query_id) != set(q):
        raise ValueError('Query IDs do not match the query folder')
    if not set(submission.winter_id).issubset(d):
        raise ValueError('Invalid winter_id')
    return True


def predict_folder(model, root, output_file, image_size, device, batch_size=32,
                   num_workers=0, embeddings_dir=None, queries_dir=None, database_dir=None):
    root = Path(root)
    queries = folder_table(queries_dir or root / 'queries', 'summer')
    database = folder_table(database_dir or root / 'database', 'winter')
    transform = image_transform(image_size)
    q_vec, _, _ = encode_images(model, queries, transform, device, batch_size, num_workers)
    d_vec, _, _ = encode_images(model, database, transform, device, batch_size, num_workers)
    if embeddings_dir:
        embeddings_dir = Path(embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(embeddings_dir / 'queries.npz', ids=queries.id.to_numpy(), vectors=q_vec)
        np.savez_compressed(embeddings_dir / 'database.npz', ids=database.id.to_numpy(), vectors=d_vec)
    indices, scores = topk_search(q_vec, d_vec, k=10)
    submission = pd.DataFrame({'query_id': queries.id.to_numpy(),
                               'winter_id': database.id.to_numpy()[indices[:, 0]]})
    validate_submission(submission, queries.id, database.id)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_file, index=False)
    return submission, (queries, database, indices, scores)


def score_mock(submission, truth_file, threshold=10):
    truth = pd.read_csv(truth_file)
    positions = dict(zip(truth.id.astype(str), truth.position.astype(int)))
    winter = submission.winter_id.map(positions)
    summer = submission.query_id.map(positions)
    if winter.isna().any() or summer.isna().any():
        raise ValueError('Missing mock ground truth for a prediction')
    errors = (winter - summer).abs()
    return {'accuracy_pm10': float((errors <= threshold).mean() * 100),
            'median_error': float(errors.median()), 'mean_error': float(errors.mean())}


def score_like_competition(submission, query_ids, database_ids, query_positions, database_positions,
                           threshold=10):
    query_ids = list(map(str, query_ids))
    valid_database = set(map(str, database_ids))
    first_prediction = submission.drop_duplicates('query_id', keep='first').set_index('query_id')
    correct = 0
    for query_id in query_ids:
        if query_id not in first_prediction.index:
            continue
        winter_id = str(first_prediction.loc[query_id, 'winter_id'])
        if winter_id not in valid_database:
            continue
        if abs(int(database_positions[winter_id]) - int(query_positions[query_id])) <= threshold:
            correct += 1
    return 100.0 * correct / len(query_ids)


def show_pairs(metadata, positions, save_file=None):
    import matplotlib.pyplot as plt
    chosen = metadata.loc[metadata.position.isin(positions)]
    fig, axes = plt.subplots(len(positions), 2, figsize=(10, 3.2 * len(positions)), squeeze=False)
    for row, position in enumerate(positions):
        for col, season in enumerate(('summer', 'winter')):
            item = chosen.loc[(chosen.position == position) & (chosen.season == season)]
            axes[row, col].axis('off')
            if item.empty:
                continue
            with Image.open(item.iloc[0].path) as image:
                axes[row, col].imshow(image.convert('RGB'))
            axes[row, col].set_title(f'{season} · {position}')
    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=140)
    return fig


def show_topk(retrieval, query_index=0, k=5, save_file=None):
    import matplotlib.pyplot as plt
    queries, database = retrieval['queries'], retrieval['database']
    indices, scores = retrieval['indices'], retrieval['scores']
    k = min(k, indices.shape[1])
    fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 3.8), squeeze=False)
    items = [queries.iloc[query_index]] + [database.iloc[idx] for idx in indices[query_index, :k]]
    for j, item in enumerate(items):
        with Image.open(item.path) as image:
            axes[0, j].imshow(image.convert('RGB'))
        axes[0, j].axis('off')
        if j == 0:
            title = f"Query summer\npos={item.position}"
        else:
            error = abs(int(item.position) - int(items[0].position))
            title = f"Top {j} · score={scores[query_index, j - 1]:.3f}\nΔ={error}"
        axes[0, j].set_title(title, fontsize=10)
    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=140)
    return fig


def show_attention(model, image_path, image_size, device, block_index=0,
                   query_indices=(0, 1, 2), save_file=None):
    import matplotlib.pyplot as plt
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    tensor = image_transform(image_size)(image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        _, maps = model(tensor, return_attention=True)
    with torch.inference_mode():
        fmap = model.aggregator.proj_c(model.backbone(tensor))
    height, width = fmap.shape[-2:]
    weights = maps[block_index][0].reshape(-1, 1, height, width)
    weights = F.interpolate(weights, size=(image.height, image.width), mode='bilinear',
                            align_corners=False).squeeze(1).cpu().numpy()
    fig, axes = plt.subplots(1, len(query_indices), figsize=(4 * len(query_indices), 3))
    axes = np.atleast_1d(axes)
    background = np.asarray(image)
    for ax, query in zip(axes, query_indices):
        heat = weights[query]
        heat = (heat - heat.min()) / (np.ptp(heat) + 1e-8)
        ax.imshow(background)
        ax.imshow(heat, alpha=0.5, cmap='magma', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Block {block_index + 1}, query {query}')
    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=140)
    return fig
