# 仓库布局与贡献

[English](../repository-layout.md) · [文档目录](README.md)

## 公开文件

| 路径 | 内容 |
| --- | --- |
| `src/deephvi/` | 维护中的包、CLI、模型、编码器、AAindex 表及 checkpoint 兼容类 |
| `configs/` | 训练、预测、历史对照及 CPU 配置 |
| `tests/` | 单元与集成测试、小型参考样例 |
| `scripts/` | 参考库下载与处理、资源打包、审查和验证工具 |
| `docs/` | 英文公开指南，中文对应版本位于 `docs/zh-CN/` |
| `data/` | 根目录中的正式资源压缩包、清单和中英文说明 |
| `.github/workflows/` | 支持版本测试与安装后 wheel 验收 |

运行 `python3 scripts/resource_packages.py extract` 恢复本地 `data/training/`、`data/models/` 等子目录。解压文件被忽略，根目录压缩包、清单和说明公开发布，详见[资源说明](resources.md)。

## 本地开发材料

开发笔记、审查报告、源码副本、打包清单和构建产物放在 `dev-docs/`。该目录由 Git 忽略，并排除在 Python 分发包之外。公开指南放在 `docs/`，实验输出放在 `runs/`。

发布前运行公开目录审查，检查已跟踪文件和新增文件。已跟踪的开发文件需要转为本地保存时，先保留本地副本，再用 `git rm --cached PATH` 移除索引条目。

## 贡献检查

先运行 `uv sync --frozen --extra dev --extra data` 安装开发依赖，再执行：

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
uv build --out-dir dev-docs/builds
```

变更公开命令或行为时同步更新中英文文档，为行为修复增加回归测试。生成的报告与实验输出保留在本地。wheel 收集 `src/deephvi`；源码分发包使用明确的公开文件允许列表和内部文件排除规则。

安装后 wheel 验收及报告说明见[验证方法](validation.md)，CI 结果可在任务输出中查看。
