# HunyuanImage 3.0 NPU MXFP8 在线量化接入 — 改动方案（调试版）

## 0. 范围

- **目标**：AR + DiT 在 NPU(Ascend) 上支持 MXFP8 **在线**量化（BF16 checkpoint，加载时量化），dense linear 走 omni 的 `NPUMxfp8OnlineLinearMethod`。
- **不量化**：MoE experts（天然 bf16，`DiffusionMXFP8Config` 不处理 FusedMoE）；MoE router gate（对齐 AR，`quant_config=None`）。
- **离线**（ascend 预量化 ckpt）：保留既有加载路径，通过 checkpoint 自动判别跳过在线注入。

> 下文行号均为 **D:\code 参考仓**版本。你工作目录（报错栈里的 `/home/k00930897/vllm-omni`）行号不同（如 process 在你那边是 `:495`，参考仓 `:486`），**用函数名/关键串定位，不要照搬行号**。

---

## 1. 两个报错的根因（均已被完整栈/源码坐实）

### 报错 A（AR）：`mxfp8 quantization is currently not supported in npu`

AR 的 `quantization: "mxfp8"` 走 **vLLM 原生** `create_engine_config`，NPU 平台校验阶段就拒绝，拿不到 omni 的 `DiffusionMXFP8Config`（`_OVERRIDES` 不经过）。

### 报错 B（DiT/AR dense 层）：`'ReplicatedLinear' object has no attribute 'weight_scale'`

完整栈关键三行：

```
mxfp8_config.py:495  replace_parameter(layer, "weight_scale", weight_scale)
layer_utils.py:25    old = getattr(mod, name)        ← 无默认值，严格版
→ AttributeError
```

- **直接原因**：在线路径 `_LazyWeightMixin.create_weights` **不注册 `weight_scale` 占位**（MRO：`NPUMxfp8OnlineLinearMethod(_LazyWeightMixin, NPUMxfp8LinearMethod)` 的 `create_weights` 解析到 `_LazyWeightMixin`，覆盖了离线兄弟里注册 weight_scale 的版本），而 `process_weights_after_loading` 要 `replace_parameter("weight_scale")`。
- **深层原因（vllm 版本差异）**：当前 NPU 环境 vllm **旧于 `fccd53258`**，`from vllm.model_executor.utils import replace_parameter` 解析到旧路径 `vllm/model_executor/layers/quantization/utils/layer_utils.py:25` 的**严格版** `getattr(mod, name)`（无默认值）。Wan2.2 测试时 vllm **≥ `fccd53258`**，用的是 `utils.py` 的**容错版** `getattr(layer, param_name, None)` + `setattr` 兜底，weight_scale 缺失也静默通过。
- **影响范围**：所有在线方法（mxfp8/mxfp4/dualscale/int8）的所有 dense linear，**不止 gate**。gate（ReplicatedLinear）只是加载顺序里最先撞上的。

---

## 2. 改动清单

### 改动 1 — 在线路径预注册 scale 占位（修复 B 核心）

**文件：`vllm_omni/quantization/mxfp8_config.py`**

**(1)** `_LazyWeightMixin` 新增模板方法（`:165`）：

```python
def _lazy_scale_param_names(self) -> tuple[str, ...]:
    """Scale parameter names that process_weights_after_loading will
    replace_parameter() into. Online subclasses override this.
    ...MRO rationale..."""
    return ()
```

**(2)** `_LazyWeightMixin.create_weights` 注册完 `weight` 后（`:236-245`）补占位注册：

```python
# Pre-register scale placeholders so that process_weights_after_loading
# can replace_parameter() them (see _lazy_scale_param_names docstring).
for name in self._lazy_scale_param_names():
    layer.register_parameter(
        name,
        torch.nn.Parameter(
            torch.empty(1, device="meta", dtype=params_dtype),
            requires_grad=False,
        ),
    )
```

**(3)** `NPUMxfp8OnlineLinearMethod` override（`:457`）：

```python
def _lazy_scale_param_names(self) -> tuple[str, ...]:
    return ("weight_scale",)
```

**文件：`vllm_omni/quantization/mxfp4_config.py`**（`_LazyWeightMixin` 从 mxfp8 import，复用占位注册）

- `NPUMxfp4OnlineLinearMethod._lazy_scale_param_names`（`:306`）→ `("weight_scale",)`
- `NPUMxfp4DualScaleOnlineLinearMethod._lazy_scale_param_names`（`:536`）→ `("weight_scale", "weight_dual_scale", "mul_scale")`

**文件：`vllm_omni/quantization/int8_config.py`**（独立 `LazyWeightMixin`，同样加了 hook + `:322` 占位注册）

- `LazyWeightMixin._lazy_scale_param_names`（`:234`）默认 `()`
- `Int8OnlineLinearMethod`（`:409`）→ `(self.int8_linear.layer_param_names[1],)`（scale 名来自 kernel 的 `layer_param_names`，不固定）
- `NPUInt8OnlineLinearMethod`（`:449`）→ `("weight_scale",)`

> 占位用普通 `torch.nn.Parameter`（非 `ModelWeightParameter`），避免被 weight loader 误填；dtype=`params_dtype`、numel=1。严格 `replace_parameter` 下：占位存在 → `getattr` 成功 → 占位 dtype(bf16) 与真实 scale(float8_e8m0fnu) 不同 → 走 `register_parameter` fallback 替换为真实 scale，占位被丢弃，无正确性影响。

---

### 改动 2 — DiT MoE gate 不量化（对齐 AR）

**文件：`vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1601`**

```python
self.gate = ReplicatedLinear(
    config.hidden_size,
    config.num_experts,
    bias=False,
    quant_config=None,   # 原 quant_config → None，对齐 AR 侧，gate 保持 bf16
    prefix=f"{prefix}.gate",
)
```

> gate 改 None 后用 `UnquantizedLinearMethod`，只注册 `weight`，与 ckpt 一致，`load_weights` 不受影响。
> MoE experts 无需改：`DiffusionMXFP8Config.get_quant_method` 对 `FusedMoE` 返回 `None` → 回退 `UnquantizedFusedMoEMethod` → experts 天然 bf16。

---

### 改动 3 — AR 在线 mxfp8 注入 `DiffusionMXFP8Config`（修复 A）

**文件：`vllm_omni/quantization/factory.py`**，新增（`:473`）：

```python
def _is_mxfp8_prequantized_checkpoint(model_path: str) -> bool:
    """True 当 ckpt config.json 声明 is_checkpoint_mxfp8_serialized（离线/ascend 预量化）。"""
    import json, os
    if not model_path or not os.path.isdir(model_path):
        return False
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, encoding="utf-8") as f:
            qc = json.load(f).get("quantization_config")
    except (OSError, ValueError):
        return False
    if not isinstance(qc, dict):
        return False
    method = _normalize_method_name(qc.get("quant_method") or qc.get("method"))
    if method != "mxfp8":
        return False
    return bool(qc.get("is_checkpoint_mxfp8_serialized"))
```

**文件：`vllm_omni/engine/stage_init_utils.py`**，`build_vllm_config` 内（`:794-834`）：`create_engine_config` 前 stash 并清空 `quantization`，之后注入：

```python
# 中和（create_engine_config 前，:804-819）
ar_online_mxfp8 = (
    current_omni_platform.is_npu()
    and _normalize_method_name(filtered_engine_args_dict.get("quantization")) == "mxfp8"
    and not _is_mxfp8_prequantized_checkpoint(omni_engine_args.model)  # BF16 ckpt 才在线
)
ar_stashed_quant = None
if ar_online_mxfp8:
    ar_stashed_quant = omni_engine_args.quantization
    omni_engine_args.quantization = None              # 避免 native 校验拒绝
    if getattr(omni_engine_args, "quantization_config", None) is not None:
        omni_engine_args.quantization_config = None
    logger.info("[stage_init] NPU AR online MXFP8: stashed native quantization=%s; ...", ar_stashed_quant)

vllm_config = omni_engine_args.create_engine_config(...)  # 不变

# 注入（create_engine_config 后，:827-834）
if ar_online_mxfp8 and ar_stashed_quant is not None:
    from vllm_omni.quantization.factory import build_quant_config
    vllm_config = replace(vllm_config, quant_config=build_quant_config("mxfp8"))
    logger.info("[stage_init] NPU AR online MXFP8: injected DiffusionMXFP8Config ...")
```

> `build_quant_config("mxfp8")` → `DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)`，其 `get_quant_method` 在 NPU 返回 `NPUMxfp8OnlineLinearMethod`。`ModelConfig.quantization` 保持 `None` 不回写，避免二次校验再次触发 NPU 拒绝。

---

### 改动 4 — deploy yaml（仅 npu 段）

`hunyuan_image3_ar.yaml` / `hunyuan_image3_dit.yaml` 的 `platforms.npu.stages[0]` 各加：

```yaml
platforms:
  npu:
    stages:
      - stage_id: 0
        # ...
        quantization: mxfp8
```

> DiT 的 `quantization` 经 `OmniDiffusionConfig.from_kwargs` 自动映射到 `quantization_config` → `build_quant_config`（与 xpu `fp8` 同机制）。

---

## 3. 调试验证步骤

### 3.1 AR 在线（NPU, BF16 ckpt）

跑 `hunyuan_image3_ar.yaml`，日志应依次出现：

```
[stage_init] NPU AR online MXFP8: stashed native quantization=mxfp8; ...
[stage_init] NPU AR online MXFP8: injected DiffusionMXFP8Config ...
Building quantization config: mxfp8
```

- ✅ 不再报 `mxfp8 quantization is currently not supported in npu`
- ✅ 不再报 `weight_scale` AttributeError
- 加载后断言：AR dense linear 的 `quant_method` 为 `NPUMxfp8OnlineLinearMethod`；MoE gate / experts 为 `Unquantized*`

### 3.2 DiT 在线（NPU, BF16 ckpt）

跑 `hunyuan_image3_dit.yaml`，日志出现 `Building quantization config: mxfp8`。

- ✅ 不再报 `'ReplicatedLinear' object has no attribute 'weight_scale'`
- 加载后断言：`qkv_proj/o_proj/gate_up_proj/down_proj` 为 `NPUMxfp8OnlineLinearMethod`；`gate` 为 `UnquantizedLinearMethod`；experts 为 `UnquantizedFusedMoEMethod`
- 出图正常

### 3.3 ascend 离线回归

用带 `is_checkpoint_mxfp8_serialized=True`（或 ascend 预量化标记）的 ckpt：

- `_is_mxfp8_prequantized_checkpoint` 返回 True → `ar_online_mxfp8=False` → 跳过注入 → 既有离线路径不受影响（走 `NPUMxfp8LinearMethod`，其 `create_weights` 本就注册 weight_scale）。

### 3.4 验证修复 B 是否真命中（因果验证，可选）

临时把 `mxfp8_config.py` 里 `_LazyWeightMixin.create_weights` 末尾的占位注册循环注释掉 → 应**复现**报错 B。这能确认占位修复是命中根因而非偶然绕过。验证完恢复。

---

## 4. 关键注意点

1. **不要回退改动 1（占位修复）**：gate 改 `quant_config=None` 只解决 gate 一层，qkv/o_proj/gate_up/down_proj 四个 dense 层仍走在线方法、仍会撞 `replace_parameter("weight_scale")`。占位修复与 gate 改动**正交，都保留**。

2. **vllm 版本差异是报错 B 的环境因素**：当前环境 vllm 旧于 `fccd53258`（严格 `replace_parameter`）；Wan2.2 测试时 ≥ `fccd53258`（容错）。所以同样代码在 Wan2.2 通过、HunyuanImage 报错。占位修复**跨版本鲁棒**——严格环境靠它绕过，将来升级到容错版也无副作用。

3. **两个 `replace_parameter`**：vllm 主仓有 `layer_utils.py`（严格，一直如此）和 `utils.py`（容错，`fccd53258` 新增）。omni `from vllm.model_executor.utils import replace_parameter` 在新旧 vllm 解析到不同实现——这是 bug 只在旧环境暴露的根因。

4. **行号差异**：你工作目录 `mxfp8_config.py` 的 process 在 `:495`、patched_weight_loader 在 `:222`，参考仓对应 `:486` / `:218`，内部结构不同（非整体偏移）。改动 1 要落在你**实际文件**的 `_LazyWeightMixin.create_weights` 末尾（`register_parameter("weight", weight)` 之后）+ `NPUMxfp8OnlineLinearMethod` 加 override。

5. **DiT gate dtype**：当前未加 `params_dtype=torch.float32`。若加载时出现 gate `weight` dtype-mismatch 告警再补（AR 侧 gate 是 fp32，DiT 未必）。

---

## 5. 改动文件汇总

| 文件 | 改动 | 对应 |
|---|---|---|
| `quantization/mxfp8_config.py` | `_LazyWeightMixin` 占位 hook + 注册；`NPUMxfp8OnlineLinearMethod` override | 改动1 |
| `quantization/mxfp4_config.py` | 两个在线方法 override | 改动1 |
| `quantization/int8_config.py` | `LazyWeightMixin` 占位 hook + 注册；两个在线方法 override | 改动1 |
| `diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py` | gate `quant_config=None` | 改动2 |
| `quantization/factory.py` | 新增 `_is_mxfp8_prequantized_checkpoint` | 改动3 |
| `engine/stage_init_utils.py` | `build_vllm_config` 中和+注入 | 改动3 |
| `deploy/hunyuan_image3_ar.yaml` / `_dit.yaml` | npu 段 `quantization: mxfp8` | 改动4 |
