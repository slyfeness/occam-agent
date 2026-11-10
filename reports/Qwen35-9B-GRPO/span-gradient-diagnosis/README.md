# 拼接诊断复核材料

- [实测报告](REPORT.md)
- [脱敏测量数据](MEASUREMENTS.json)
- [诊断源码](probe/)

源码从报告记录的私有实验提交提取。公开版将模型位置改为 `OCCAM_STEP0_MODEL`，提取共享评分和 FSDP 辅助函数，并包含后来用于完成首层分片对照的 dtype 修正。诊断方法和归约公式沿用实测版本。公开版的打包检查覆盖 Python 语法及合成输入单测，GPU 数值来自报告中的原始作业。

GPU 复核需要自行提供有权限使用的本地 step 0 模型和三份批次文件。批次是包含 `trajectories` 数组的 JSON，每条有 `id`、`advantage` 和 `segments`。每个 segment 包含 `prompt_ids`、`completion_ids`、`vllm_logprobs`。主探针读取 128 条轨迹，按已记录的规则选取 8 条。两个历史输入需按顺序提供 180554 和 181174 的轨迹。历史四处边界会独立核对。

实测环境为 4×A800 80GB、PyTorch 2.10.0、Transformers 5.5.1、Flash Linear Attention 0.6.0、causal-conv1d 1.7.0。以下命令从本目录运行，输出路径请使用新文件或新目录。

```bash
export OCCAM_STEP0_MODEL=/path/to/local/step0-model
mkdir -p results
python -m unittest discover -s tests
python -m torch.distributed.run --standalone --nproc_per_node=4 -m probe.probe_span_cause \
  --batch "$MAIN_BATCH" --history "$HISTORY_A" "$HISTORY_B" \
  --output results --source-commit "$(git rev-parse HEAD)"
python -m torch.distributed.run --standalone --nproc_per_node=4 -m probe.probe_span_backward_path \
  --batch "$MAIN_BATCH" --forward results/FORWARD.json \
  --output results/BACKWARD_PATH.json --source-commit "$(git rev-parse HEAD)"
python -m torch.distributed.run --standalone --nproc_per_node=4 -m probe.probe_span_gate \
  --batch "$MAIN_BATCH" --forward results/FORWARD.json \
  --output results/GATE.json --source-commit "$(git rev-parse HEAD)"
```

测量数据按统计和匿名边界事件发布。原生 token 序列、题目、回答、原始轨迹和模型资产存放在私有实验存储。
