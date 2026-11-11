"""Render the requested per-step report, CSV and curves from sealed step receipts."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from .stability import read_json, write_json


def row_for(receipt, audit):
    train, api = receipt['train'], receipt['api']
    row = {k: receipt.get(k) for k in (
        'step', 'reward_mean', 'group_all_correct', 'group_all_wrong', 'group_mixed', 'group_unscored',
        'scored_trajectories', 'reward_excluded_count', 'normal_count', 'truncation_rate',
        'empty_or_invalid_rate', 'repetition_rate', 'mean_tool_calls', 'mean_rounds',
        'wall_seconds', 'generation_seconds', 'judge_tail_seconds', 'local_judge_seconds',
        'train_wall_seconds', 'sync_seconds', 'other_seconds', 'memory_utilization_colocated',
        'colocated_peak_mib', 'retrieval_calls', 'retrieval_local_only',
        'retrieval_rejected_before_search', 'retrieval_timing_unmatched',
        'retrieval_wall_seconds', 'retrieval_fraction_of_generation')}
    row.update({k: train.get(k) for k in ('optimizer_step', 'stop_reason', 'loss', 'entropy', 'grad_norm',
                                          'action_tokens', 'tis_clip_fraction', 'forward_backward_seconds',
                                          'optimizer_seconds', 'checkpoint_seconds', 'export_seconds',
                                          'adam_state_step_min', 'adam_state_step_max')})
    row.update({f'tis_abs_{k}': (train.get('tis_absolute_logprob_error') or {}).get(k)
                for k in ('mean', 'p99', 'max')})
    row.update({f'api_{k}': api.get(k) for k in (
        'wall_seconds', 'mean_latency_seconds', 'p95_latency_seconds', 'retries', 'failures')})
    row['api_model'] = ' / '.join(api.get('served_models', []))
    actors = receipt.get('actor_stats') or [{} for _ in range(4)]
    peaks = receipt.get('train_peak_nvml_mib') or [None] * 4
    for i, (actor, peak) in enumerate(zip(actors, peaks, strict=True)):
        row[f'actor{i}_prefix_hit_rate'] = actor.get('prefix_hit_rate')
        row[f'actor{i}_preemptions'] = actor.get('num_preemptions_total')
        row[f'gpu{i}_train_peak_mib'] = peak
    for status in ('MODEL_FINAL_STOP', 'CONTEXT_LIMIT_STOP', 'GENERATION_LENGTH_STOP',
                   'SAFETY_CEILING_STOP', 'EMPTY_MODEL_STOP', 'FORMAT_ERROR'):
        row[status.lower()] = receipt['termination'].get(status, 0)
    row['oom_retries'] = len(receipt.get('oom_fallbacks', []))
    row['policy_synced'] = receipt.get('policy_synced', False)
    for key in ('output_tokens', 'context_tokens_sum', 'longest_context_tokens', 'api_peak_concurrency'):
        row[key] = audit.get(key)
    for key in ('retrieval_event_wall', 'retrieval_server_queue', 'retrieval_client_to_search'):
        row[f'{key}_mean_seconds'] = audit.get(key, {}).get('mean_seconds')
        row[f'{key}_p95_seconds'] = audit.get(key, {}).get('p95_seconds')
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}


def plot(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [row['step'] for row in rows]
    fig, grid = plt.subplots(4, 2, figsize=(10.4, 10.2), constrained_layout=True)
    plots = [('Mean reward', [('reward_mean', 'Reward')]), ('Policy entropy', [('entropy', 'Entropy, nats')]),
             ('Gradient norm', [('grad_norm', 'Before clipping')]),
             ('TIS log-prob gap', [('tis_abs_mean', 'Mean'), ('tis_abs_p99', 'P99'), ('tis_abs_max', 'Max')]),
             ('TIS clipping', [('tis_clip_fraction', 'Clipped fraction')]),
             ('Trajectory truncation', [('truncation_rate', 'Truncated fraction')]),
             ('Mean tool calls', [('mean_tool_calls', 'Calls per trajectory')]),
             ('Per-step time', [('wall_seconds', 'Entire step'), ('generation_seconds', 'Rollout')])]
    for axes, (title, series) in zip(grid.flat, plots):
        for field, label in series:
            axes.plot(steps, [row.get(field) for row in rows], marker='o', label=label)
        axes.set(title=title, xlabel='Step', xticks=steps)
        if title == 'TIS log-prob gap':
            values = [row[field] for field, _ in series for row in rows if row[field] is not None]
            if values and min(values) > 0:
                axes.set_yscale('log')
        axes.grid(alpha=.25)
        axes.legend()
    fig.savefig(directory / 'CURVES.svg')
    plt.close(fig)


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--slurm-state')
    parser.add_argument('--failure-cause')
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    status = read_json(args.run / 'STATUS.json')
    status.update(slurm_state=args.slurm_state, failure_cause=args.failure_cause)
    plan = read_json(args.run / 'RUN_PLAN.json')
    receipts = []
    for folder in sorted(args.run.glob('step-*')):
        result = folder / 'STEP_RESULT.json'
        if result.is_file():
            receipts.append(read_json(result))
            continue
        rollout = sorted(folder.glob('rollout-*/ROLLOUT_RESULT.json'))
        if rollout:
            partial = read_json(rollout[-1])
            step = int(folder.name.split('-')[1])
            trained = args.run / f'trained-{step:02d}.json'
            train = read_json(trained) if trained.is_file() else {}
            if args.failure_cause:
                train.setdefault('stop_reason', args.failure_cause)
            agreement = folder / 'JUDGE_AGREEMENT.json'
            partial.update(step=step, train=train, agreement=read_json(agreement) if agreement.is_file() else None)
            receipts.append(partial)
    audit = read_json(args.audit) if args.audit else {}
    rows = [row_for(record, audit.get('steps', {}).get(str(record['step']), {})) for record in receipts]
    args.destination.mkdir(mode=0o700, exist_ok=False)
    if rows:
        with (args.destination / 'STEPS.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, args.destination)
    agreement = next((row['agreement'] for row in receipts if row.get('agreement')), None)
    stats = {'status': status, 'agreement': agreement, 'rows': rows,
             'plan_sha256': plan['pool_sha256'], 'seed': plan['seed'],
             'heldout_reason': plan['heldout_reason'], 'checkpoint_audit': audit.get('checkpoints')}
    if args.private:
        stats['query_ids'] = plan['training']
    write_json(args.destination / 'RESULT.json', stats)
    lines = ['# Qwen3.5-9B 全参 GRPO 十步稳定性', '',
             f"作业 `{status['job']}`。状态 `{status['state']}`，已执行 {status['optimizer_steps']} 次 optimizer.step。", '',
             '| 步 | 奖励均值 | 熵（nat） | 裁剪前梯度范数 | Loss | TIS P99 | TIS 截断比例 | 轨迹截断率 | 平均调用数 | 整步（秒） |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fields = ('step', 'reward_mean', 'entropy', 'grad_norm', 'loss', 'tis_abs_p99', 'tis_clip_fraction',
              'truncation_rate', 'mean_tool_calls', 'wall_seconds')
    lines += ['| ' + ' | '.join(
        f'{row[k]:.3e}' if k in ('loss', 'tis_clip_fraction') and row[k] is not None else fmt(row[k])
        for k in fields) + ' |' for row in rows]
    walls = [row['wall_seconds'] for row in rows if row['wall_seconds'] is not None]
    lines += ['', f'完成同步的整步平均耗时为 {fmt(mean(walls) if walls else None, 1)} 秒。',
              'API 判分墙钟包含与生成重叠的时段，整步加总只计生成结束后的判分等待。检索占比按工具调用区间并集除以生成墙钟。']
    if agreement is not None:
        lines += ['', f"第 1 步本地判分尝试 {agreement['attempted']} 条，可比较 {agreement['comparable']} 条，"
                  f"一致 {agreement['agreements']} 条。正常终止子集一致 {agreement['normal_agreements']}/{agreement['normal_comparable']}。"]
    else:
        lines += ['', '第 1 步双判分的一致率缺少完整测量。']
    stop = next((row['stop_reason'] for row in rows if row['stop_reason']), None)
    if stop:
        lines += ['', f'提前停止原因 `{stop}`。']
    synchronized = sum(bool(row['optimizer_step'] and row['policy_synced']) for row in rows)
    if status['state'] == 'complete' and synchronized == 10:
        lines += ['', '连续十步均完成更新与权重同步。可冻结本次实测的训练配置用于同规模复测。']
    else:
        lines += ['', f'暂缓冻结。已完成更新并同步的步数为 {synchronized}/10。']
    lines += ['',
              '为控制本轮稳定性试验的时长，跳过可选的第 0 步和第 10 步留出评测。']
    if rows:
        lines += ['', '[逐步 CSV](STEPS.csv) · [指标曲线](CURVES.svg) · [脱敏数据](RESULT.json)']
    if args.private:
        lines += ['', '## 冻结题目 ID', '']
        measured = {row['step'] for row in rows}
        lines.extend(f'- 第 {i} 步（{"已生成" if i in measured else "计划"}），' + ', '.join(group)
                     for i, group in enumerate(plan['training'], 1))
    (args.destination / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
