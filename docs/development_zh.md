# 开发说明

本项目的开发目标是维护单一生产推理架构：`fastest` profile + native paged-KV + vLLM-like online batching。

## 本地检查

```bash
uv run ruff check .
uv run ty check .
uv run python -m py_compile qwen3_tts_ov/*.py scripts/*.py examples/python/*.py
uv run pytest -q
```

文档和示例的静态检查在 `tests/test_docs_examples.py` 中覆盖：JSON/JSONL 可解析、Python example 可解析、README/docs/examples 的相对链接存在。

## 构建 native backend

```bash
uv run python scripts/build_native_codegen.py
```

构建产物在 `native/build/`，不进入 git。

## 构建生产 IR

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

`build-fastest` 只构建生产图集合，不再提供历史诊断 graph-set。

## 性能与质量

性能矩阵：

```bash
uv run python scripts/benchmark_prompt_batch_matrix.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --batch-sizes 1,2,4,8,16 \
  --prompt-lengths short,medium,long,xlong \
  --scenarios offline,online \
  --runs 3
```

三模式架构 gate：

```bash
uv run python scripts/evaluate_single_arch_gate.py \
  --server-url http://127.0.0.1:17860 \
  --modes voice_design,custom_voice,voice_clone \
  --runs 3 \
  --concurrency 1,2,4,8
```

需要和原始 PyTorch 链路对照时，使用 `scripts/evaluate_prefill_quality.py --candidate-path runtime` 或 `scripts/verify_long_autoregressive_parity.py`。默认 reference 要求 CUDA/XPU，不静默回退 CPU。

## Release

GitHub Actions 在普通提交上执行检查和构建，在 `v*` tag 上发布 release。源码仓库不提交模型、IR、outputs 或 native build 产物。

本地打包：

```bash
uv run python scripts/package_release.py --help
```

版本更新 checklist：

1. 更新 `pyproject.toml` 中的 `version`。
2. 更新 `docs/release_zh.md`、`docs/release.md`、根 README 中的版本号、下载文件名和 release 链接。
3. 新增 `docs/releases/vX.X.X.md` release notes。
4. 更新 `docs/releases/README.zh-CN.md` 和 `docs/README.md` 中的 release notes 入口。
5. 确认 `scripts/package_release.py`、`scripts/package_ir.py` 默认版本或 workflow 输入与目标版本一致。
6. 运行文档链接检查和必要 smoke 后打 `git tag vX.X.X`，再推送 tag。

提交前不要把以下目录加入 git：`models/`、`openvino/`、`openvino_full/`、`outputs/`、`native/build/`、`dist/release/`、`.venv/`、`.uv-cache/`。
