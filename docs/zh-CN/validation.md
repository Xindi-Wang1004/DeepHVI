# 验证方法

[English](../validation.md) · [文档目录](README.md)

在仓库根目录执行以下命令，检查代码、资源和安装包。

## 常规检查

运行 `uv sync --frozen --extra dev --extra data` 后，在仓库根目录执行：

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
```

公开目录审查同时覆盖已跟踪文件及未忽略新文件，拒绝内部文档、正式压缩包之外的模型文件和其他生成产物。测试涵盖记录解析与去重、划分与逐折追踪、高斯融合、指标、特征兼容性、归档处理、续训状态、检索和 CLI 流程。

参考库集成测试使用本地 HTTP 样例和 hash 演示编码器，检查下载、索引复用、故障恢复、训练记录、分阶段继承和 checkpoint 检索，并对比启用与省略参考准备时的模型参数、配对/划分指纹及分类输出。

## 真实资源

先恢复压缩包，再检查完整性及服务器数值对照：

```bash
python3 scripts/resource_packages.py extract
uv run python scripts/verify_resource_layout.py --resources-dir data --config-dir configs
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python scripts/verify_server_resources.py \
  --resources-dir data --out dev-docs/validation/server
```

服务器检查比较缓存及重算嵌入、独立参考编码输出、边界 token 长度、新序列编码和 CLI 预测，数值容差为 `atol=2e-5, rtol=2e-5`。报告记录环境及实测误差。

## 安装后 wheel 验收

```bash
python3 scripts/verify_release.py --python 3.12 --out dev-docs/validation/release
```

需要 PATH 中可运行 `uv`，并为额外一份解压资源及环境预留空间。脚本审查公开文件、导出独立副本、恢复资源、安装锁定依赖、构建并安装 wheel，然后确认导入来自 `site-packages`。随后执行测试、资源审查、真实 ESM-2 服务器预测，以及 18 对样本、三折、各阶段一轮的小型训练。

每次运行使用新的 `--out` 目录，临时文件、日志、构建产物和报告会一起保存在其中。CI 对 Python 3.10 和 3.12 运行测试，并在独立任务中执行安装后验收。

检查在 Linux CPU 环境运行，覆盖 hash 示例、真实 ESM-2 预测与训练，以及断点恢复后的参数一致性。
