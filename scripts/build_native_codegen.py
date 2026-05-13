#!/usr/bin/env python3
"""Build the optional native Qwen3-TTS codegen shared library."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys


def find_first(root: pathlib.Path, names: tuple[str, ...]) -> pathlib.Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def resolve_build_paths():
    import openvino
    import openvino_genai
    import openvino_tokenizers

    repo = pathlib.Path(__file__).resolve().parents[1]
    ov_root = pathlib.Path(openvino.__file__).resolve().parent
    genai_root = pathlib.Path(openvino_genai.__file__).resolve().parent
    include_dir = ov_root / "include"
    lib_dir = ov_root / "libs"
    genai_include_dir = repo / "third_party" / "openvino.genai" / "src" / "cpp" / "include"
    tokenizers_ext_path = pathlib.Path(openvino_tokenizers._ext_path).resolve()

    if os.name == "nt":
        openvino_lib = find_first(ov_root, ("openvino.lib", "openvino.dll"))
        genai_lib = find_first(genai_root, ("openvino_genai.lib", "openvino_genai.dll"))
    elif sys.platform == "darwin":
        openvino_lib = find_first(lib_dir, ("libopenvino.dylib",))
        genai_lib = find_first(genai_root, ("libopenvino_genai.dylib",))
    else:
        openvino_lib = find_first(lib_dir, ("libopenvino.so.2610", "libopenvino.so"))
        genai_lib = find_first(genai_root, ("libopenvino_genai.so.2610", "libopenvino_genai.so"))

    for label, path in (
        ("OpenVINO include dir", include_dir),
        ("OpenVINO GenAI include dir", genai_include_dir),
        ("OpenVINO C++ library", openvino_lib),
        ("OpenVINO GenAI C++ library", genai_lib),
    ):
        if path is None or not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return {
        "repo": repo,
        "source_dir": repo / "native" / "qwen3_tts_ov_genai",
        "source": repo / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_codegen.cpp",
        "cli_source": repo / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_cli.cpp",
        "openvino_include": include_dir,
        "openvino_lib": pathlib.Path(openvino_lib),
        "openvino_lib_dir": lib_dir,
        "genai_include": genai_include_dir,
        "genai_lib": pathlib.Path(genai_lib),
        "genai_root": genai_root,
        "tokenizers_ext": tokenizers_ext_path,
    }


def shared_library_name() -> str:
    if os.name == "nt":
        return "qwen3_tts_ov_genai.dll"
    if sys.platform == "darwin":
        return "libqwen3_tts_ov_genai.dylib"
    return "libqwen3_tts_ov_genai.so"


def run_direct_build(args, paths: dict[str, pathlib.Path], output_dir: pathlib.Path) -> None:
    if os.name == "nt":
        raise SystemExit("direct native build is not supported on Windows; use --backend cmake")
    output = output_dir / shared_library_name()
    cli_output = output_dir / ("qwen3_tts_ov_native_cli.exe" if os.name == "nt" else "qwen3_tts_ov_native_cli")
    flags = ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    cmd = [
        args.cxx,
        "-std=c++17",
        "-fPIC",
        "-shared",
        *flags,
        f"-I{paths['openvino_include']}",
        str(paths["source"]),
        str(paths["openvino_lib"]),
        f"-I{paths['genai_include']}",
        str(paths["genai_lib"]),
        f"-Wl,-rpath,{paths['genai_root']}",
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{paths['openvino_lib_dir']}",
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
        f"-I{paths['openvino_include']}",
        f"-I{paths['source_dir']}",
        f"-DDEFAULT_OPENVINO_TOKENIZERS_PATH=\"{paths['tokenizers_ext']}\"",
        str(paths["cli_source"]),
        str(output),
        f"-Wl,-rpath,{output_dir}",
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{paths['genai_root']}",
        f"-Wl,-rpath,{paths['openvino_lib_dir']}",
        "-o",
        str(cli_output),
    ]
    print(" ".join(cli_cmd), flush=True)
    subprocess.run(cli_cmd, check=True)
    print(cli_output, flush=True)


def run_cmake_build(args, paths: dict[str, pathlib.Path], output_dir: pathlib.Path) -> None:
    cmake = args.cmake or shutil.which("cmake")
    if not cmake:
        raise SystemExit("cmake not found; install CMake or use --backend direct on Linux")
    build_dir = output_dir / "cmake-build"
    cmd = [
        cmake,
        "-S",
        str(paths["source_dir"]),
        "-B",
        str(build_dir),
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}",
        f"-DOPENVINO_INCLUDE_DIR={paths['openvino_include']}",
        f"-DOPENVINO_LIB={paths['openvino_lib']}",
        f"-DOPENVINO_GENAI_INCLUDE_DIR={paths['genai_include']}",
        f"-DOPENVINO_GENAI_LIB={paths['genai_lib']}",
        f"-DOPENVINO_TOKENIZERS_PATH={paths['tokenizers_ext']}",
        f"-DBUILD_NATIVE_CLI={'OFF' if args.no_cli else 'ON'}",
    ]
    if args.cmake_generator:
        cmd.extend(["-G", args.cmake_generator])
    if args.debug:
        cmd.append("-DCMAKE_BUILD_TYPE=Debug")
    else:
        cmd.append("-DCMAKE_BUILD_TYPE=Release")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    build_cmd = [cmake, "--build", str(build_dir), "--config", "Debug" if args.debug else args.config]
    print(" ".join(build_cmd), flush=True)
    subprocess.run(build_cmd, check=True)

    output = output_dir / shared_library_name()
    if not output.exists() and os.name == "nt":
        matches = sorted(output_dir.rglob("qwen3_tts_ov_genai.dll"))
        if matches:
            shutil.copy2(matches[0], output)
    if not output.exists():
        raise FileNotFoundError(f"native library was not produced: {output}")
    print(output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="native/build")
    parser.add_argument("--backend", default="auto", choices=["auto", "direct", "cmake"])
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--cmake", default=None)
    parser.add_argument("--cmake-generator", default=None)
    parser.add_argument("--config", default="Release")
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

    paths = resolve_build_paths()
    output_dir = (paths["repo"] / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = args.backend
    if backend == "auto":
        backend = "cmake" if os.name == "nt" else "direct"
    if backend == "cmake":
        run_cmake_build(args, paths, output_dir)
    else:
        run_direct_build(args, paths, output_dir)


if __name__ == "__main__":
    main()
