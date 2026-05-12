#!/usr/bin/env python3
"""Build the optional native Qwen3-TTS codegen shared library."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="native/build")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-cli", action="store_true", help="Only build the shared library, not the native CLI.")
    parser.add_argument(
        "--no-genai-link",
        action="store_true",
        help="Deprecated: the native Qwen3-TTS pipeline requires libopenvino_genai.",
    )
    args = parser.parse_args()
    if args.no_genai_link:
        raise SystemExit("--no-genai-link is no longer supported; the native pipeline requires libopenvino_genai")

    import openvino
    import openvino_genai
    import openvino_tokenizers

    repo = pathlib.Path(__file__).resolve().parents[1]
    source = repo / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_codegen.cpp"
    cli_source = repo / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_cli.cpp"
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "libqwen3_tts_ov_genai.so"
    cli_output = output_dir / "qwen3_tts_ov_native_cli"

    ov_root = pathlib.Path(openvino.__file__).resolve().parent
    include_dir = ov_root / "include"
    lib_dir = ov_root / "libs"
    lib_openvino = lib_dir / "libopenvino.so.2610"
    if not lib_openvino.exists():
        raise FileNotFoundError(f"OpenVINO C++ library not found: {lib_openvino}")
    genai_root = pathlib.Path(openvino_genai.__file__).resolve().parent
    genai_include_dir = repo / "third_party" / "openvino.genai" / "src" / "cpp" / "include"
    lib_openvino_genai = genai_root / "libopenvino_genai.so.2610"
    tokenizers_ext_path = pathlib.Path(openvino_tokenizers._ext_path).resolve()
    if not genai_include_dir.exists():
        raise FileNotFoundError(
            f"OpenVINO GenAI headers not found: {genai_include_dir}; clone the submodule/repo first"
        )
    if not lib_openvino_genai.exists():
        raise FileNotFoundError(f"OpenVINO GenAI C++ library not found: {lib_openvino_genai}")

    flags = ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    genai_args = [
        f"-I{genai_include_dir}",
        str(lib_openvino_genai),
        f"-Wl,-rpath,{genai_root}",
    ]
    cmd = [
        args.cxx,
        "-std=c++17",
        "-fPIC",
        "-shared",
        *flags,
        f"-I{include_dir}",
        str(source),
        str(lib_openvino),
        *genai_args,
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(output),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(output, flush=True)
    if args.no_cli:
        return

    cli_cmd = [
        args.cxx,
        "-std=c++17",
        *flags,
        f"-I{include_dir}",
        f"-I{repo / 'native' / 'qwen3_tts_ov_genai'}",
        f"-DDEFAULT_OPENVINO_TOKENIZERS_PATH=\"{tokenizers_ext_path}\"",
        str(cli_source),
        str(output),
        f"-Wl,-rpath,{output_dir}",
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{genai_root}",
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(cli_output),
    ]
    print(" ".join(cli_cmd), flush=True)
    subprocess.run(cli_cmd, check=True)
    print(cli_output, flush=True)


if __name__ == "__main__":
    main()
