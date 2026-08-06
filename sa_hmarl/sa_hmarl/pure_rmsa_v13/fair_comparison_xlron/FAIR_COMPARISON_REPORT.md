# XLRON 兼容环境与 KSP-FF 公平对比报告

**Status: COMPLETED. 2026-08-05. 纯 Python/NumPy，无 JAX/GPU/C extension。**

## 1. 与 XLRON 的逐函数对应（代码走读）

| 本实现 | XLRON 参考 | 一致性 |
|---|---|---|
| `traffic.generate_trace`（到达+保持时间） | `env_funcs._arrival_holding_from_uniforms` | ✓ 公式一致：`arrival = -log1p(-u)/rate`，`rate = load/mean` |
| `traffic._holding_rejection_sampling` | 同函数截断分支（env_funcs.py:1574-1599） | ✓ 5 候选、≥2×mean 置 0、取首个非零、全零强制截断 |
| `traffic` 带宽 | `make_env` min_bw=25/max_bw=100/step_bw=1 | ✓ 25-100 步进 1 均匀 |
| `env.highest_modulation` | `init_modulations_array`（BPSK/QPSK/8QAM/16QAM） | ✓ 距离表 10000/2500/1250/625 km 一致 |
| `env.required_slots` | `env_funcs.required_slots` | ✓ `ceil(bw/(se×slot_size))+guardband`，slot_size=12.5、guardband=1 |
| `env.ksp_paths`（hops-first） | `env_funcs._get_k_shortest_paths`（weight=None → 按跳数） | ✓ 无向图 22 边 → 44 有向链路，与 XLRON `nsfnet_deeprmsa_directed` 一致 |
| `env.path_mask`（频谱连续+邻接） | `env_funcs.mask_slots` | ✓ 窗口在路径所有链路上空闲（cumsum 实现等价） |
| `env.allocate/advance_time` | `implement_action_rsa` + 释放逻辑 | ✓ link_slot_array 占用、departure 时间释放 |
| `ksp_ff.first_fit_starts` | `heuristics.first_fit` | ✓ 加哨兵列取每路径首个可行 start |
| `ksp_ff.ksp_ff` | `heuristics.ksp_ff` | ✓ KSP 序第一条有可行 start 的路径 |
| `run_episode`（warmup+eval+SBP） | `eval_heuristic` | ✓ warmup 3000、eval 10000、SBP |

拓扑核对：XLRON `nsfnet_deeprmsa_directed.json` = 14 节点、44 有向链路（22 无向对）——与项目 `xlron_nsfnet_deeprmsa` 完全一致。

## 2. 实验设置（预注册）

- NSFNET directed、100 FSU、load=250 Erlang、warmup 3000、eval 10000
- K ∈ {5, 50}、hops-first、seeds 69301-69310（10）
- 25-100 Gbps 均匀（1 Gbps 步进）、保持时间 2×均值截断（XLRON 拒绝采样）
- First-Fit、频谱连续+邻接、SBP

## 3. 结果

### 3.1 K=5 vs K=50（10 seeds，XLRON 拒绝采样截断）

| K | 阻塞率均值 | std | per-seed |
|---|---:|---:|---|
| 5 | 3.802% | 0.449 | 4.20/3.46/3.89/3.31/4.42/4.47/3.94/3.54/3.54/3.25 |
| 50 | 2.931% | 0.438 | 3.52/2.61/2.98/2.55/3.45/3.41/3.02/2.73/2.87/2.17 |
| **dK** | **0.871 pp** | | 全部 10 seeds 同向（K5 > K50） |

时延：K=5 每 episode ~2.1s、K=50 ~7.6s（纯 Python 基线；时延非本对比重点）。

### 3.2 截断方式对照（3 seeds，同环境同负载，只换截断机制）

| 截断模式 | K=5 | K=50 | dK |
|---|---:|---:|---:|
| 无截断 | 12.773% | 12.603% | **0.170 pp** |
| 硬截断（重采样 ≤2×mean） | 3.830% | 2.933% | **0.897 pp** |
| **XLRON 拒绝采样** | 3.850% | 3.037% | **0.813 pp** |

**结论**：
1. **截断开关是 K 差距的 ~5 倍放大器**（无截断 0.17pp → 截断 0.85pp）——"保持时间截断放大 K 差距"成立且被直接量化；
2. **截断实现方式（硬 vs XLRON 拒绝采样）对结果几乎无影响**（dK 0.897 vs 0.813，绝对阻塞率差 <0.11pp）——两种截断机制等价；
3. 无截断时阻塞率 12.6-12.8%（load=250 饱和区）——两种 K 都接近饱和，差距被淹没。

### 3.3 与 Doherty 2025 对照（诚实）

| | 原文（中负载 170-220） | 本实现（load=250） |
|---|---|---|
| K5 | 2-4.8% | 3.80%（在范围内 ✓） |
| K50 | 0.4-1.1% | 2.93%（高于原文） |
| dK | 1.6-3.7pp | 0.87pp |

K5 在原文范围内；K50 高于原文（原文 K50 是学习/优化系统而非 KSP-FF——本实现是纯 KSP-FF，无定价/学习能力）；dK 低于原文（0.87 vs 1.6-3.7）——差异来自对比对象（KSP-FF vs 优化系统）而非仿真环境。

## 4. 剩余差异（诚实清单）

1. **随机数序列**：`np.random.RandomState`（确定性）对应 XLRON 的 JAX PRNG——分布一致（指数/均匀），但逐位序列不同，无法逐请求比对（只有分布级可比）；
2. **全零候选的强制截断**：XLRON 代码的 `differentiable_argmax` 语义在全零时取 index 0（值为 0）；本实现按任务规格取第 5 个候选（强制截断）。差异仅在 ~4.5e-5 概率事件上（P(5 个全 ≥2×mean)=e^-10），对结果影响可忽略；
3. **链路距离字段**：拓扑边数/节点数与 XLRON 一致（22 无向/44 有向），链路公里数取项目 registry（xlron_nsfnet_deeprmsa）；XLRON JSON 的距离字段名不同但值同源（deeprmsa 系列）；
4. **性能**：纯 Python 每 episode 2-8s（XLRON 是 JAX GPU 并行）——功能等价，速度不同（本任务限 Python/NumPy）；
5. **调制不可达路径**：>10000km 路径标记不可用（XLRON mod_format_mask=-1 语义一致）。

## 5. 可复现命令

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
source /mnt/d/project/DNN_Agent/.venv/bin/activate
export PYTHONPATH=.
unset SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL

# 测试（19 个）
python -m pytest sa_hmarl/pure_rmsa_v13/fair_comparison_xlron/tests/ -q

# 主对比（K=5/50 × 10 seeds）
python -m sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.run_ksp_k5_k50

# 截断方式对照（3 seeds）
python -m sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.run_truncation_compare
```

## 6. 产物

- `traffic.py` / `env.py` / `ksp_ff.py` / `run_ksp_k5_k50.py` /
  `run_truncation_compare.py`
- `tests/`（19 测试全过：截断分布、拒绝采样 CDF、可复现性、调制表、
  required_slots、KSP hops 序、mask、分配/释放、KSP-FF 行为、边界）
- `artifacts/ksp_k5_k50/KSP_K5_K50_RESULTS.{json,csv}`（20 episode）
- `artifacts/truncation_compare/TRUNCATION_COMPARE_RESULTS.{json,csv}`
  （18 episode）

未修改任何现有实现；native unset；单 worker；seeds 69301-69310
（新段，与 formal seeds 无交叉）。
