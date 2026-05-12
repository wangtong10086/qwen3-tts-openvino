# Devtools

这里保存非生产主线的实验和历史脚本。生产部署默认只使用根目录 README 中的 `fastest` profile。

- `bench/`: 历史 benchmark、profiling 和量化对照脚本。
- `legacy/`: 旧导出/推理入口，仅用于回看迁移过程。

这些脚本可能暴露 fp16、SDPA、RMS、ll-v2、fast-cache、no-cache 等实验路径；它们不作为默认运行方式，也不作为实时性能验收标准。
