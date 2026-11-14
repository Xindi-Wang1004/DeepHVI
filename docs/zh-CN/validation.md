# 验证方法

[English](../validation.md) · [文档目录](README.md)

验证覆盖软件行为、资源一致性和已发布模型兼容性，不代表论文完整实验指标已复现。请针对实际使用的修订版本运行下列命令，历史开发报告不能替代当前检查。

## 常规检查

运行 `uv sync --frozen --extra dev --extra data` 后，在仓库根目录执行：

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
```

公开目录审查同时覆盖已跟踪文件及未忽略新文件，拒绝内部文档、正式压缩包之外的模型文件和其他生成产物。测试涵盖记录解析与去重、划分与逐折追踪、高斯融合、指标、特征兼容性、归档处理、续训状态、检索和 CLI 流程。

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

每次运行请指定新的 `--out`：已有非空目录会被拒绝。临时文件、日志、构建产物和报告均保存在该忽略目录内。CI 对 Python 3.10 和 3.12 运行测试，并在独立任务中执行安装后验收。

## 如何解读

hash 后端用于无蛋白语言模型的流程检查；小型真实 ESM 实验验证配套资源上的训练与预测可运行。CPU 续训测试在固定环境中比较连续训练和恢复后的参数。完整训练指标、CUDA 可复现性、Windows/macOS 行为及所有最低依赖组合需要单独验证。原数据库快照缺失也限制了数据重建，详见[数据说明](data.md)。
