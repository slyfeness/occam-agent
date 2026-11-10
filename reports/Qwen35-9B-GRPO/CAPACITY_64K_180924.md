# 4×A800 Qwen3.5-9B 64K 续训容量验收

Slurm 180924 正常完成。使用 `BACKWARD_POST`，第一轮完成一次真实 GRPO/AdamW 更新，四个 rank 的 Adam 步数均从 1 到 2。没有 OOM，也没有触发选择性激活卸载。裁剪前梯度范数 1.58985，四个 rank 均观察到参数变化。该次更新只用于容量验收，恢复源仍是 180648 的 step 1。

恢复了既有四个 Actor 和 AdamW 分片。每卡常驻 FP32 Adam 动量为 16.68 GiB。批次保留原 16 条轨迹、189 次模型调用、56,025 个动作 token、组内优势、完整历史和动作掩码。给最长调用的原生 prompt 插入 10,097 个模型可见的合成上下文 token，和原 55,439 token 一起构成完整 65,536 token。新增 token 属于 prompt，直接 loss 掩码为 0。rank 0 在累积梯度后完成了该条 64K 输入的完整前向和反向。本测试测的是一条 64K 微批次混在原批次中的训练容量，样本是已判分批次的容量重放。

| Rank | NVML 整卡采样峰值 GiB | PyTorch allocated 峰值 GiB | PyTorch reserved 峰值 GiB |
| --- | ---: | ---: | ---: |
| 0 | 79.23 | 75.32 | 78.24 |
| 1 | 79.25 | 68.34 | 78.34 |
| 2 | 78.99 | 68.01 | 78.21 |
| 3 | 79.01 | 68.62 | 78.19 |

三列峰值各自独立统计。NVML 按秒采样，rank 顺序由 CUDA 设备 UUID 核对。完整作业 23 分 02 秒，checkpoint 恢复 6 分 24 秒，首个 old log-prob 评分到 optimizer step 结束约 11 分 42 秒。rank 0 的 64K 微批次耗时为 old log-prob 22.67 秒、训练前向 5.82 秒、反向 40.29 秒，合计约 68.78 秒。真实 AdamW step 约 0.81 秒。

运行源码为 `a522a8c`。原始证据保存在私有实验存储中。

其中 `COMPLETE.json` 汇总四个 rank 的更新和峰值，`FIXTURE.json` 绑定原始批次及插入位置，`backward_post/TIMING-rank*.jsonl` 和 `backward_post/GPU.csv` 保存计时与采样。
