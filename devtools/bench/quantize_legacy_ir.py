import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OV_TELEMETRY_DISABLE", "1")

import nncf
import numpy as np
import openvino as ov


def add_suffix(path: str, suffix: str) -> str:
    item = Path(path)
    return f"{item.stem}{suffix}{item.suffix}"


def input_name(input_port, index: int):
    names = input_port.get_names()
    return input_port.get_any_name() if names else index


def make_dataset(files: list[Path], model: ov.Model, subset_size: int) -> nncf.Dataset:
    input_names = [input_name(port, index) for index, port in enumerate(model.inputs)]
    selected = files[:subset_size]

    def transform(path: Path):
        with np.load(path) as data:
            return {name: data[f"input_{index}"] for index, name in enumerate(input_names)}

    return nncf.Dataset(selected, transform)


def quantize_model(
    ir_dir: Path,
    source_graph: str,
    target_graph: str,
    calibration_files: list[Path],
    subset_size: int,
    preset,
    model_type,
    force: bool,
) -> None:
    target = ir_dir / target_graph
    if target.exists() and target.with_suffix(".bin").exists() and not force:
        print(f"exists {target}; skipping", flush=True)
        return
    if not calibration_files:
        raise ValueError(f"no calibration files for {source_graph}")

    started = time.time()
    print(f"quantizing {source_graph} -> {target_graph} with {min(len(calibration_files), subset_size)} samples", flush=True)
    core = ov.Core()
    model = core.read_model(ir_dir / source_graph)
    dataset = make_dataset(calibration_files, model, subset_size)
    quantized = nncf.quantize(
        model,
        dataset,
        preset=preset,
        target_device=nncf.TargetDevice.GPU,
        subset_size=min(len(calibration_files), subset_size),
        model_type=model_type,
    )
    ov.save_model(quantized, target, compress_to_fp16=False)
    print(f"saved {target} in {time.time() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--calibration-dir", default="outputs/calibration_nocache")
    parser.add_argument("--variant", default="int8_ptq")
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--preset", default="mixed", choices=["mixed", "performance"])
    parser.add_argument("--model-type", default="transformer", choices=["transformer", "none"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    calibration_dir = Path(args.calibration_dir)
    manifest_path = ir_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    preset = {
        "mixed": nncf.QuantizationPreset.MIXED,
        "performance": nncf.QuantizationPreset.PERFORMANCE,
    }[args.preset]
    model_type = nncf.ModelType.TRANSFORMER if args.model_type == "transformer" else None

    graphs = manifest["graphs"]
    variant_graphs = {}
    jobs = [
        ("talker", "talker_no_cache", graphs["talker"]),
        ("subcode_greedy", "subcode_greedy", graphs["subcode_greedy"]),
    ]
    for manifest_key, calibration_prefix, source_graph in jobs:
        target_graph = add_suffix(source_graph, f"_{args.variant}")
        files = sorted(calibration_dir.glob(f"{calibration_prefix}_*.npz"))
        quantize_model(ir_dir, source_graph, target_graph, files, args.subset_size, preset, model_type, args.force)
        variant_graphs[manifest_key] = target_graph

    variants = manifest.setdefault("graph_variants", {})
    variants[args.variant] = {
        "precision": f"ptq_{args.preset}_int8",
        "graphs": variant_graphs,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
