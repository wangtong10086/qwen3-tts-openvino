# 大文件与产物策略

本仓库采用源码仓库策略，不提交模型权重、OpenVINO IR、生成音频或本地虚拟环境。

## 不提交的目录

```text
models/          原始 Qwen3-TTS 模型权重
openvino/        推荐导出的 OpenVINO IR 目录
openvino_full/   旧本地完整 IR 目录
outputs/         生成音频、benchmark 输出和 profile
.venv/           本地 Python 虚拟环境
.uv-cache/       uv 缓存
native/build/    native C++ 编译产物
build/release/   PyInstaller/CMake 中间产物
dist/release/    app/IR release 包
*.egg-info/      本地 editable install 元数据
```

## 为什么不提交 OpenVINO IR

一个完整 VoiceDesign IR 可能超过几十 GB，单个 `.bin` 文件也可能超过 GitHub 普通 Git 文件限制。把这些文件提交到源码仓库会导致 clone、review、CI 和配额都变得不可用。

推荐做法：

1. 在源码仓库保存导出脚本和 manifest 约定。
2. 每个开发者在本地按 `docs/export_zh.md` 重新导出 IR。
3. 需要共享 IR 时，使用对象存储、内部制品库或 GitHub Release 附件，而不是普通 Git commit。

最终用户 release 分成 app 包和 IR 包。app 包包含可执行 server、OpenVINO runtime 和 native 库；IR 包包含 `openvino/<model_type>/manifest.json` 及其引用的 `.xml/.bin`。OpenVINO compile cache 不进入任何 release 包。

## 检查命令

提交前确认没有误加入大文件：

```bash
git ls-files | rg '^(models|openvino|openvino_full|outputs)/|\\.bin$|\\.wav$'
```

该命令应无输出。

查看本地仓库对象体积：

```bash
git count-objects -vH
```
