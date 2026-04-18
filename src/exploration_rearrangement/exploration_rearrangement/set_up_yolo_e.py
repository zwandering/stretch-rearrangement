"""Export the smallest YOLOE variant (yoloe-11s-seg) with text prompts baked in.

The text prompts come from `config/objects.yaml` in this package, so the
runtime detector node and the exported artifact share one source of truth.

Typical usage (from workspace root, with the same venv that runs the node):

    python -m exploration_rearrangement.set_up_yolo_e
    # or, after colcon build + source install/setup.bash:
    ros2 run exploration_rearrangement set_up_yolo_e

Supported formats (``--format``):
    engine     TensorRT .engine      - NVIDIA GPU only (fastest when available).
    onnx       ONNX .onnx            - CPU or GPU via ONNXRuntime.
    openvino   *_openvino_model/ dir - Intel CPU / iGPU (best for Stretch 3 NUC).
    ncnn       *_ncnn_model/ dir     - Pure-CPU, small footprint.

For CPU-only targets (e.g. Stretch 3), prefer ``--format openvino --device cpu``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

import yaml

# Default = smallest YOLOE available upstream (~20 MB). Auto-downloaded on first use.
DEFAULT_MODEL = 'yoloe-11s-seg.pt'

# Maps --format → suffix or directory-name suffix that ultralytics emits.
# Values ending with '/' are directories (export creates a folder).
_FORMAT_ARTIFACT = {
    'engine':   '.engine',
    'onnx':     '.onnx',
    'openvino': '_openvino_model/',
    'ncnn':     '_ncnn_model/',
    'torchscript': '.torchscript',
    'mnn':      '.mnn',
    'paddle':   '_paddle_model/',
    'tflite':   '.tflite',
}

# Formats that can't use GPU.
_CPU_ONLY_FORMATS = {'openvino', 'tflite'}


def _find_objects_yaml() -> Path:
    """Locate objects.yaml relative to this file, then fall back to CWD."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / 'config' / 'objects.yaml',
        Path.cwd() / 'src' / 'exploration_rearrangement' / 'config' / 'objects.yaml',
        Path.cwd() / 'config' / 'objects.yaml',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f'objects.yaml not found; looked in: {[str(c) for c in candidates]}'
    )


def _load_prompts(yaml_path: Path) -> List[str]:
    """Flatten every per-object prompt into the ordered list YOLOE expects."""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    prompts: List[str] = []
    for entry in cfg.get('objects', []):
        entry_prompts = entry.get('prompts')
        if entry_prompts:
            prompts.extend(entry_prompts)
        else:
            prompts.append(entry['name'].replace('_', ' '))
    if not prompts:
        raise ValueError(f'No prompts found in {yaml_path}')
    return prompts


def _parse_imgsz(s: str):
    if 'x' in s.lower():
        h, w = s.lower().split('x')
        return (int(h), int(w))
    return int(s)


def _clean_prior_artifacts(weights_dir: Path, stem: str) -> None:
    """Remove any prior export output next to the weights, file or directory."""
    for artifact in _FORMAT_ARTIFACT.values():
        path = weights_dir / f'{stem}{artifact.rstrip("/")}'
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f'[set_up_yolo_e] removed stale {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Export YOLOE with baked prompts.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--objects-yaml', default=None)
    parser.add_argument(
        '--format', default='engine',
        choices=sorted(_FORMAT_ARTIFACT.keys()),
        help='Export target backend.',
    )
    parser.add_argument('--imgsz', default='640', help='"640" or "HxW" (e.g. 640x640)')
    parser.add_argument('--device', default='0',
                        help='GPU index (e.g. "0") or "cpu". CPU-only formats force cpu.')
    parser.add_argument('--half', dest='half', action='store_true', default=True,
                        help='FP16 weights (engine/onnx/openvino). Default on.')
    parser.add_argument('--no-half', dest='half', action='store_false')
    parser.add_argument('--int8', action='store_true',
                        help='INT8 quantization (needs --data for calibration).')
    parser.add_argument('--data', default=None,
                        help='Calibration dataset yaml, required by --int8.')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace (GB); ignored for other formats.')
    parser.add_argument('--dynamic', action='store_true',
                        help='Dynamic input shapes (onnx/openvino).')
    args = parser.parse_args()

    from ultralytics import YOLOE  # imported lazily so unit tests don't need it

    yaml_path = Path(args.objects_yaml) if args.objects_yaml else _find_objects_yaml()
    prompts = _load_prompts(yaml_path)
    print(f'[set_up_yolo_e] prompts = {prompts}')
    print(f'[set_up_yolo_e] format  = {args.format}')

    # Force device=cpu for CPU-only formats to avoid confusing ultralytics warnings.
    device_str = args.device
    if args.format in _CPU_ONLY_FORMATS and device_str != 'cpu':
        print(f'[set_up_yolo_e] {args.format} is CPU-only; overriding --device → cpu')
        device_str = 'cpu'
    device_arg: object = int(device_str) if device_str.isdigit() else device_str

    if args.int8 and not args.data:
        parser.error('--int8 requires --data <calibration.yaml>')

    model = YOLOE(args.model)
    model.set_classes(prompts, model.get_text_pe(prompts))

    weights_dir = Path(args.model).resolve().parent
    stem = Path(args.model).stem
    _clean_prior_artifacts(weights_dir, stem)

    export_kwargs = dict(
        format=args.format,
        device=device_arg,
        imgsz=_parse_imgsz(args.imgsz),
        half=args.half and not args.int8,   # can't combine
        int8=args.int8,
        dynamic=args.dynamic,
    )
    if args.format == 'engine':
        export_kwargs['workspace'] = args.workspace
    if args.data:
        export_kwargs['data'] = args.data

    exported = model.export(**export_kwargs)
    print(f'[set_up_yolo_e] exported → {exported}')


if __name__ == '__main__':
    main()
