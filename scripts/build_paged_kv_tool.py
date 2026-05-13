#!/usr/bin/env python3
"""Build the Qwen3-TTS OpenVINO paged-KV graph conversion tool."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="native/build")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import openvino

    repo = pathlib.Path(__file__).resolve().parents[1]
    source = repo / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_paged_kv_tool.cpp"
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_ov_paged_kv_tool"

    ov_root = pathlib.Path(openvino.__file__).resolve().parent
    include_dir = ov_root / "include"
    lib_dir = ov_root / "libs"
    lib_openvino = lib_dir / "libopenvino.so.2610"
    if not lib_openvino.exists():
        raise FileNotFoundError(f"OpenVINO C++ library not found: {lib_openvino}")

    flags = ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    cmd = [
        args.cxx,
        "-std=c++17",
        *flags,
        f"-I{include_dir}",
        str(source),
        str(lib_openvino),
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(output),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
