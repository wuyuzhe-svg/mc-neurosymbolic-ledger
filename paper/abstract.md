# 【已存档 2026-07-18】AAAI 摘要包(决定不投;GitHub repo 为最终产物,措辞素材保留于此)

## 标题候选

1. **Teaching a World Model to Place One Block: Counterfactual Supervision for
   Sparse Interaction Actions and a Neuro-Symbolic Inventory Ledger**
   (主推:具体、有画面感,"one block/1 bit" 呼应金样本)
2. Sparse Interaction Actions in Small-Data World Models: Event-Level
   Counterfactual Supervision and an In-Loop Symbolic Ledger
3. From Pixels to Provenance: Grounding Inventory Accounting in a Generative
   World Model via Counterfactual Event Supervision

## Abstract(~195 词,AAAI 上限 200 内)

Interactive world models learn dense actions (locomotion, camera) readily, but
sparse interaction actions — placing a block, breaking one — routinely fail to
become causal: under reconstruction training on 62 hours of Minecraft gameplay,
the `place` action receives ~0.2% of the gradient and the resulting model
places blocks regardless of the key being pressed. We show that event-level
counterfactual supervision — a differential margin between action-ON and
action-OFF denoising trajectories, applied only in the high-noise band where
event identity is decided — turns a single input bit into reliable causality:
12/12 held-out windows respond to the keypress and stay silent without it,
with a paired probe score of 0.994 vs 0.015. On top of this causal generator
we close a neuro-symbolic loop: latent-space probes detect interaction events
and identify the interacted object (80.4% over 16 block types, transferring to
generated rollouts), and a deterministic inventory ledger both records the
consequences and vetoes actions the inventory cannot support. Along the way we
document a measured chain of eleven failure modes — teacher leakage, warm-init
contamination, behavioral echo, anchor-free counterfactuals — that we argue
constitutes a reusable supervision-engineering methodology for sparse events
in small-data world models. Weights, evaluation assets, and demos are released.

## 摘要中文对照(自查用)

密集动作(移动/视角)容易学,稀疏交互动作(放置/破坏)在重建训练下拿到
~0.2% 梯度、按不按键都放置。事件级反事实监督(ON/OFF 去噪轨迹差分 margin,
只施加在决定事件的高噪段)把 1 bit 输入变成可靠因果:12/12 窗口按键响应/
不按静默,成对探针分 0.994 vs 0.015。其上闭合神经-符号回路:latent 探针
检测事件并识别交互对象(16 类 80.4%,迁移到生成态),确定性账本记账并对
库存不支持的动作行使否决。同时给出 11 条带测量的失败模式链作为可复用的
稀疏事件监督工程方法论。权重与评测素材全部开源。

## 三条贡献(camera-ready 措辞)

1. **Event-level counterfactual supervision** for sparse interaction actions:
   differential (ON−OFF) margin in the high-noise decision band; we show why
   absolute suppression of the OFF branch is unsafe (learned smearing) and why
   differential constraints are immune to teacher leakage.
2. **A neuro-symbolic inventory loop**: latent probes (event when + object
   what) drive a deterministic ledger that records mine/place transitions and
   vetoes infeasible actions — guarantees the neural write-path cannot give
   under small data (held-injector gate weight 0.002; AR place non-transfer).
3. **A measured failure-mode chain** (11 entries, each with an experiment):
   reconstruction dilution, teacher leakage, warm-init contamination,
   encoding mismatch, behavioral echo, anchor-free counterfactual cheating,
   CFG dose-response axis, first-frame anchor dead zone, direction asymmetry
   of sparse events — offered as reusable supervision-engineering methodology.
