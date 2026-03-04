import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'logs/plots'))
METRICS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'logs/metrics'))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'logs/checkpoints'))

COLORS = {
    'branch_type': {
        'cnnA': '#e6194b',
        'cnnAspp': '#3cb44b',
        'cnnB': '#4363d8',
        'cnnBspp': '#f58231',
    },
    'dim': {
        '102': '#e6194b',
        '227': '#3cb44b',
        '323': '#4363d8',
    },
    'channels': {
        1: '#e6194b',
        3: '#4363d8',
    },
    'degrees': {
        1: '#e6194b',
        3: '#4363d8',
    },
    'weight_config': {
        'scratch': '#e6194b',
        'frozen': '#3cb44b',
        'fine-tuned': '#4363d8',
    },
}

COMPARISONS = [
    {
        'attr': 'branch_type',
        'get_value': lambda m: m.branch_type,
        'title': 'Branch Type',
        'name': 'by_branch',
    },
    {
        'attr': 'channels',
        'get_value': lambda m: m.channels,
        'title': 'Channels',
        'name': 'by_channels',
    },
    {
        'attr': 'degrees',
        'get_value': lambda m: m.degrees,
        'title': 'Degrees',
        'name': 'by_degrees',
    },
    {
        'attr': 'weight_config',
        'get_value': lambda m: m.weight_config,
        'title': 'Weight Config',
        'name': 'by_weights',
    },
]


@dataclass
class ModelResult:
  image_height: int
  image_width: int
  frames: int
  load_weights: bool
  train_weights: bool
  channels: int
  degrees: int
  branch_type: str
  epochs: int = 200
  train_loss: list = field(default_factory=list)
  val_loss: list = field(default_factory=list)
  total_time_sec: float = 0.0
  test_loss: float = float('inf')
  test_mae: float = float('inf')
  test_quaternion_loss: float = float('inf')
  disk_size_mb: float = float('nan')

  @property
  def dim(self) -> str:
    return str(self.image_height)

  @property
  def weight_config(self) -> str:
    if self.load_weights and self.train_weights:
      return 'fine-tuned'
    if self.load_weights:
      return 'frozen'
    return 'scratch'

  @property
  def best_val(self) -> float:
    return min(self.val_loss) if self.val_loss else float('inf')

  @property
  def time_per_100_epochs(self) -> float:
    return (self.total_time_sec / self.epochs) * 100 if self.epochs > 0 else 0.0

  @property
  def stem(self) -> str:
    return (
        f'{self.image_height}_{self.image_width}_{self.frames}_'
        f'{int(self.load_weights)}_{int(self.train_weights)}_'
        f'{self.channels}_{self.degrees}_{self.branch_type}'
    )


def available_dims(results: List[ModelResult]) -> List[str]:
  return sorted({m.dim for m in results})


def _slices(
    results: List[ModelResult],
    dims: List[str],
) -> List[tuple[str, List[ModelResult]]]:
  """(label, subset) pairs: 'All' first, then one per dim."""
  entries: List[tuple[str, List[ModelResult]]] = [('All', results)]
  for d in dims:
    entries.append((f'{d}px', [m for m in results if m.dim == d]))
  return entries


def _nanmean_curves(curves: List[list]) -> np.ndarray:
  max_len = max(len(c) for c in curves)
  padded = np.full((len(curves), max_len), np.nan)
  for i, c in enumerate(curves):
    padded[i, :len(c)] = c
  return np.nanmean(padded, axis=0)


def _save(fig: plt.Figure, save_name: str) -> None:
  path = os.path.join(OUTPUT_DIR, f'{save_name}.png')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'  ✓ {path}')


def _make_grid(
    n_panels: int,
    panel_w: float,
    panel_h: float,
    sharey: bool = True,
) -> tuple[plt.Figure, list[plt.Axes], list[plt.Axes]]:
  """
  Build a 2-column grid tall enough to hold n_panels subplots.
  Returns (fig, flat_active_axes, flat_all_axes).
  flat_all_axes includes any empty padding cell so the caller can hide it.
  """
  n_cols = 2
  n_rows = (n_panels + 1) // 2          # ceiling division

  fig, axes = plt.subplots(
      n_rows, n_cols,
      figsize=(panel_w * n_cols, panel_h * n_rows),
      sharey=sharey,
      squeeze=False,                        # always 2-D array
  )

  flat_all = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]
  flat_active = flat_all[:n_panels]

  # Hide the spare cell when n_panels is odd
  for ax in flat_all[n_panels:]:
    ax.set_visible(False)

  return fig, flat_active, flat_all


def load_all(metrics_dir: str = METRICS_DIR) -> List[ModelResult]:
  results = []
  for fname in sorted(os.listdir(metrics_dir)):
    if not fname.endswith('.json'):
      continue
    parts = Path(fname).stem.split('_')
    with open(os.path.join(metrics_dir, fname)) as fp:
      data = json.load(fp)
    test = data.get('test', {})
    epochs = len(data.get('quaternion_loss', []))
    results.append(ModelResult(
        image_height=int(parts[0]),
        image_width=int(parts[1]),
        frames=int(parts[2]),
        load_weights=bool(int(parts[3])),
        train_weights=bool(int(parts[4])),
        channels=int(parts[5]),
        degrees=int(parts[6]),
        branch_type=parts[7],
        epochs=epochs,
        train_loss=data.get('quaternion_loss', []),
        val_loss=data.get('val_quaternion_loss', []),
        total_time_sec=data.get('total_time_sec', 0.0),
        test_loss=test.get('loss', float('inf')),
        test_mae=test.get('mae', float('inf')),
        test_quaternion_loss=test.get('quaternion_loss', float('inf')),
    ))
  print(f"Loaded {len(results)} model results from '{metrics_dir}/'")
  return results


def load_checkpoint_stats(
    results: List[ModelResult],
    checkpoints_dir: str = CHECKPOINTS_DIR,
) -> None:
  checkpoint_map: dict[str, Path] = {}
  if os.path.isdir(checkpoints_dir):
    for folder in os.scandir(checkpoints_dir):
      if folder.is_dir():
        checkpoint_map[folder.name] = Path(folder.path)

  for m in results:
    folder = checkpoint_map.get(m.stem)
    if folder is None:
      matches = [p for k, p in checkpoint_map.items() if k.startswith(m.stem)]
      folder = matches[0] if len(matches) == 1 else None
    if folder is None:
      continue
    keras_files = sorted(folder.glob('*.keras'))
    if not keras_files:
      continue
    m.disk_size_mb = (
        sum(f.stat().st_size for f in keras_files) / len(keras_files) / (1024 ** 2)
    )


def grid_box_by_attribute(
    results: List[ModelResult],
    dims: List[str],
    attr: str,
    get_value: Callable[[ModelResult], Any],
    metric_fn: Callable[[ModelResult], float],
    sup_title: str,
    ylabel: str,
    save_name: str,
) -> None:
  slices = _slices(results, dims)
  palette = COLORS[attr]
  fig, axes, _ = _make_grid(len(slices), panel_w=6, panel_h=4.5, sharey=True)

  for ax, (col_label, subset) in zip(axes, slices):
    groups: dict = {}
    for m in subset:
      v = metric_fn(m)
      if not np.isnan(v) and v >= 0:
        groups.setdefault(get_value(m), []).append(v)

    keys = [k for k in palette if k in groups]

    ax.set_title(col_label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    if not keys:
      ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
              ha='center', va='center', color='grey', fontsize=10)
      ax.set_xticks([])
      continue

    bp = ax.boxplot(
        [groups[k] for k in keys],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white',
                       markeredgecolor='black', markersize=5),
        medianprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp['boxes'], [palette[k] for k in keys]):
      patch.set_facecolor(color)
      patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([str(k) for k in keys], fontsize=9,
                       rotation=15, ha='right')

  # Left-column axes get the y-label
  for ax in axes[::2]:
    ax.set_ylabel(ylabel, fontsize=10)

  fig.suptitle(sup_title, fontsize=14, fontweight='bold')
  fig.tight_layout()
  _save(fig, save_name)


def grid_mean_curves_by_attribute(
    results: List[ModelResult],
    dims: List[str],
    attr: str,
    get_value: Callable[[ModelResult], Any],
    sup_title: str,
    save_name: str,
) -> None:
  """
  2-column grid of validation-loss curve panels.
  """
  slices = _slices(results, dims)
  palette = COLORS[attr]
  fig, axes, _ = _make_grid(len(slices), panel_w=7, panel_h=4.5, sharey=True)

  for ax, (col_label, subset) in zip(axes, slices):
    groups: dict = {}
    for m in subset:
      if not m.val_loss:
        continue
      val = get_value(m)
      color = palette.get(val, '#999999')
      groups.setdefault(val, []).append(m.val_loss)
      ax.plot(range(1, len(m.val_loss) + 1), m.val_loss,
              color=color, alpha=0.2, linewidth=0.7)

    for key, color in palette.items():
      if key not in groups:
        continue
      mean = _nanmean_curves(groups[key])
      ax.plot(range(1, len(mean) + 1), mean,
              color=color, linewidth=2.2, label=str(key))

    ax.set_title(col_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3)

    if not groups:
      ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
              ha='center', va='center', color='grey', fontsize=10)

  # Left-column axes get the y-label
  for ax in axes[::2]:
    ax.set_ylabel('Validation Quaternion Loss', fontsize=10)

  # Single shared legend on the last active axis
  handles = [
      mlines.Line2D([], [], color=c, linewidth=2.2, label=str(k))
      for k, c in palette.items()
  ]
  axes[-1].legend(
      handles=handles,
      title=attr.replace('_', ' ').title(),
      fontsize=9,
      loc='upper right',
  )

  fig.suptitle(sup_title, fontsize=14, fontweight='bold')
  fig.tight_layout()
  _save(fig, save_name)


def grid_loss_overview(
    results: List[ModelResult],
    dims: List[str],
    save_name: str = 'loss_overview',
) -> None:
  """
  2-column grid of train/val mean-curve panels.
  """
  slices = _slices(results, dims)
  fig, axes, _ = _make_grid(len(slices), panel_w=7, panel_h=4.5, sharey=True)

  for ax, (col_label, subset) in zip(axes, slices):
    train_curves, val_curves = [], []
    for m in subset:
      if len(m.train_loss) < 100:
        continue
      train_curves.append(m.train_loss)
      val_curves.append(m.val_loss)
      ax.plot(range(1, len(m.train_loss) + 1), m.train_loss,
              color='C0', alpha=0.2, linewidth=0.6)
      ax.plot(range(1, len(m.val_loss) + 1), m.val_loss,
              color='C1', alpha=0.2, linewidth=0.6, linestyle='--')

    for curves, color, ls in [(train_curves, 'C0', '-'), (val_curves, 'C1', '--')]:
      if curves:
        mean = _nanmean_curves(curves)
        ax.plot(range(1, len(mean) + 1), mean,
                color=color, linewidth=2.2, linestyle=ls)

    ax.set_title(col_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3)

    if not train_curves:
      ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
              ha='center', va='center', color='grey', fontsize=10)

  for ax in axes[::2]:
    ax.set_ylabel('Quaternion Loss', fontsize=10)

  handles = [
      mlines.Line2D([], [], color='C0', linewidth=2.2, linestyle='-', label='Train'),
      mlines.Line2D([], [], color='C1', linewidth=2.2, linestyle='--', label='Validation'),
  ]
  axes[-1].legend(handles=handles, fontsize=9, loc='upper right')

  fig.suptitle('Train & Validation Loss by Dimension', fontsize=14, fontweight='bold')
  fig.tight_layout()
  _save(fig, save_name)


def main():
  results = load_all()
  load_checkpoint_stats(results)
  dims = available_dims(results)
  print(f'\nGenerating plots… (dims found: {dims})\n')

  print('— Validation loss curves —')
  for comp in COMPARISONS:
    grid_mean_curves_by_attribute(
        results, dims,
        attr=comp['attr'],
        get_value=comp['get_value'],
        sup_title=f"Val Loss by {comp['title']}  (All vs per-Dimension)",
        save_name=f"{comp['name']}_curves",
    )

  print('\n— Best validation loss box plots —')
  for comp in COMPARISONS:
    grid_box_by_attribute(
        results, dims,
        attr=comp['attr'],
        get_value=comp['get_value'],
        metric_fn=lambda m: m.best_val,
        sup_title=f"Best Val Loss by {comp['title']}  (All vs per-Dimension)",
        ylabel='Best Validation Loss',
        save_name=f"{comp['name']}_box",
    )

  print('\n— Training time box plots —')
  for comp in COMPARISONS:
    grid_box_by_attribute(
        results, dims,
        attr=comp['attr'],
        get_value=comp['get_value'],
        metric_fn=lambda m: m.time_per_100_epochs,
        sup_title=f"Time / 200 Epochs by {comp['title']}  (All vs per-Dimension)",
        ylabel='Time per 200 Epochs (sec)',
        save_name=f"time_{comp['name']}_box",
    )

  print('\n— Checkpoint disk size —')
  grid_box_by_attribute(
      results, dims,
      attr='branch_type',
      get_value=lambda m: m.branch_type,
      metric_fn=lambda m: m.disk_size_mb,
      sup_title='Checkpoint Disk Size by Branch Type  (All vs per-Dimension)',
      ylabel='Average Checkpoint Size (MB)',
      save_name='disk_size_by_branch_box',
  )

  print('\n— Train / val loss overview —')
  grid_loss_overview(results, dims, save_name='loss_overview')

  print(f"\nAll plots saved to '{OUTPUT_DIR}/'")


if __name__ == '__main__':
  main()
