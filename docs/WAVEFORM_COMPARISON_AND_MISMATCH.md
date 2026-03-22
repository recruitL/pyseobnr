# 波形对比与 NR Mismatch 使用说明

本文说明如何将**自定义模型**（如 `SEOBNRv5HM_nonspin`）与现有模型做波形对比，以及如何计算与数值相对论（NR）的 mismatch。

---

## 一、与现有 EOB 模型对比（EOB vs EOB）

使用脚本 `pyseobnr/auxiliary/sanity_checks/EOB_matches.py`，可对两个 EOB 模型在随机参数点上做逐 mode 的 unfaithfulness（mismatch）。

### 1. 自定义模型 vs SEOBNRv5HM

```bash
cd /path/to/pyseobnr
python -m pyseobnr.auxiliary.sanity_checks.EOB_matches \
  --model-1-name SEOBNRv5HM_nonspin \
  --model-2-name SEOBNRv5HM \
  --points 1000 \
  --q-max 10 \
  --chi-max 0.5 \
  --n-cpu 8 \
  --name my_nonspin_vs_v5HM
```

- `model_1` 为你的 non-spin 模型（内部 chi=0）
- `model_2` 为 align-spin 的 SEOBNRv5HM
- 会生成 `my_nonspin_vs_v5HM_SEOBNRv5HM_nonspin_SEOBNRv5HM{ell}{m}.dat` 及各 mode 的 mismatch 统计

### 2. 仅比较 (2,2) 或少量点（快速测试）

可把 `--points` 调小（如 50），或后续在脚本里把 `mode_list` 改为 `[(2,2)]` 做单 mode 测试。

### 3. 在 Python 里直接算单次 mismatch

若只想对一组参数算“自定义模型 vs 现有模型”的 mismatch，可直接用 `UnfaithfulnessModeByModeLAL`，两个输入都是已运行过的 `Model` 实例（带 `t`, `waveform_modes`, `delta_T`）：

```python
import numpy as np
from pyseobnr.generate_waveform import generate_modes_opt
from pyseobnr.auxiliary.sanity_checks.metrics import UnfaithfulnessModeByModeLAL

q, chi1, chi2 = 2.0, 0.0, 0.0
omega0 = 0.015

# 你的模型 (SEOBNRv5HM_nonspin)
_, _, model_nonspin = generate_modes_opt(
    q, 0.0, 0.0, omega0, debug=True, approximant="SEOBNRv5HM_nonspin"
)

# 现有模型 (SEOBNRv5HM)
_, _, model_v5 = generate_modes_opt(
    q, chi1, chi2, omega0, debug=True, approximant="SEOBNRv5HM"
)

unf = UnfaithfulnessModeByModeLAL(settings={"debug": False, "masses": np.arange(10, 210, 20)})
mismatch_22 = unf(model_nonspin, model_v5, ell=2, m=2)
print(f"(2,2) unfaithfulness: {mismatch_22}")
```

---

## 二、与 NR 的 Mismatch（NR vs EOB）

使用脚本 `pyseobnr/auxiliary/sanity_checks/NR_mismatches.py`，对多支 NR 波形逐 case、逐 mode 算与给定 EOB 的 mismatch。

### 1. NR 数据格式与 cases 文件

- **SXS**：需要 SXS 目录结构，且 cases 文件为 CSV，其中一列为 NR 目录路径（脚本中会拼成 `row[5] + "/rhOverM_Asymptotic_GeometricUnits_CoM.h5"`）。详见 `get_NR_paths()`。
- 依赖：`scri`、`sxs`，以及 `NRModel_SXS`（见 `pyseobnr/auxiliary/external_models`）。

### 2. 用自定义模型做 NR mismatch

```bash
python -m pyseobnr.auxiliary.sanity_checks.NR_mismatches \
  --cases-file /path/to/your/spinning_calibs.csv \
  --model SEOBNRv5HM_nonspin \
  --n-cpu 8
```

- `--model SEOBNRv5HM_nonspin` 表示 EOB 侧使用你的 non-spin 模型（内部 chi=0）；NR 侧仍用 NR 的 q、chi1、chi2 等（用于取 NR 的 omega0 等）。
- 输出：参数文件 + 各 mode 的 mismatch 文件（如 `mismatch_SEOBNRv5HM_nonspin22.dat` 等），以及可选画图。

### 3. 仅 non-spin NR 与自定义模型对比

若只关心 nonspinning NR，可在准备 CSV 时只保留 chi1≈0, chi2≈0 的 case；脚本不变，仍用 `--model SEOBNRv5HM_nonspin`。

---

## 三、用到的工具与接口

| 用途           | 脚本/类 | 说明 |
|----------------|--------|------|
| EOB vs EOB     | `EOB_matches.py` | 两 EOB 模型在随机 (q,chi1,chi2) 上逐 mode mismatch；已支持 `SEOBNRv5HM_nonspin` |
| NR vs EOB      | `NR_mismatches.py` | 多支 NR 与指定 EOB 的逐 mode mismatch；已支持 `SEOBNRv5HM_nonspin` |
| 单次评估       | `UnfaithfulnessModeByModeLAL` | 任意两个满足 `Model` 接口的实例（含自定义模型）逐 mode unfaithfulness |

自定义模型只要通过 `generate_modes_opt(..., approximant="SEOBNRv5HM_nonspin")` 得到带 `t`, `waveform_modes`, `delta_T` 的 Model，即可与 NR 或其它 EOB 一样参与上述流程。

---

## 四、极化与应变层面的 match（可选）

若需要**应变** \(h_+/h_\times\) 上的 match（含时间/相位优化），可参考：

- `pyseobnr/auxiliary/sanity_checks/aligned_matches_v5PHM.py`：用 `generate_v5PHM_waveform` 生成应变，再用 `pycbc.filter.matchedfilter.optimized_match` 算 match。
- 对你的模型：先用 `generate_modes_opt(..., approximant="SEOBNRv5HM_nonspin", debug=True)` 得到 model，再从其 `model.waveform_modes` 与 `model.t` 合成 \(h_+, h_\times\)，然后与上述脚本同样方式做 FFT、PSD 和 `optimized_match`。

如需把“自定义 approximant”接入 `generate_v5PHM_waveform` 的接口，需要在 `generate_waveform.py` 里为应变生成路径增加对 `SEOBNRv5HM_nonspin` 的分支（与现有 `SEOBNRv5HM` 类似），再复用同一套 match 代码。
