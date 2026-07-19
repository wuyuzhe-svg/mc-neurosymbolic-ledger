# A World Model That Maintains Its Inventory

A 1.3B interactive video model with a working backpack: mining credits it (+1, with the
item's identity), placing debits it (−1), and when it hits zero the ledger vetoes the
action. Trained on 62 hours of Minecraft gameplay.

[![Weights & Assets](https://img.shields.io/badge/HuggingFace-weights_%2B_eval_assets-yellow)](https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger)
[![Base](https://img.shields.io/badge/base-Wan2.1--1.3B_(Matrix--Game--2)-blue)](https://github.com/SkyworkAI/Matrix-Game)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)

<table>
<tr>
<td width="50%" align="center" valign="top">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/clip_mine_sand_3s.gif" alt="Mine sand: ledger credits +1"/>
  <br/><b>Attack sand → +1</b>. The model digs under an attack keypress. At the moment
  the block disappears, the what-head reads <code>sand@1.00</code> and the ledger credits
  +1 sand to the hotbar.
</td>
<td width="50%" align="center" valign="top">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/clip_attack_grass_c47.gif" alt="Attack a grass block: ledger credits +1"/>
  <br/><b>Attack grass → +1</b>. Different scene and item. The raised grass block is dug
  open; the event probe fires (0.57), pixel settlement confirms the removal, the what-head
  reads <code>grass_block@0.99</code>, and the ledger credits +1.
</td>
</tr>
</table>
<table>
<tr>
<td width="50%" align="center" valign="top">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/clip_place_dirt_minus1.gif" alt="Place dirt: ledger debits to zero"/>
  <br/><b>Place dirt → −1</b>. With 1 dirt in stock the place key is allowed through;
  the block appears, the place probe confirms (1.0), and the ledger debits to zero.
</td>
<td width="50%" align="center" valign="top">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/clip_place_cobble_minus1.gif" alt="Place cobblestone: ledger debits to zero"/>
  <br/><b>Place cobblestone → −1</b>. Same protocol, different item and scene. The block
  fills the wall gap at the crosshair, the probe confirms (1.0), and the ledger debits
  to zero.
</td>
</tr>
</table>

<p align="center">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/demo_strip_4rows.png" width="100%" alt="Frame strips of the four demos: attack sand, attack grass, place dirt, place cobblestone"/>
  <br/>
  <em>Frame strips of the four demos above, one row each. The hotbar overlay is rendered
  from ledger state, not generated pixels; it flips exactly when the sensors report the
  credit or debit. A full three-act cycle (mine +1 → place −1 → veto at zero) is in
  <a href="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/economy_demo.mp4">economy_demo.mp4</a>.</em>
</p>

## Why I made this

I got into interactive video models the way most people do: the demos look like magic,
you walk around a generated world and it holds together. Then you try to *do* anything —
break a block, pick something up, build — and the spell breaks. Nothing persists. The
hotbar is wallpaper: items pop in and out with no arithmetic behind them, and nothing
stops the model from placing a block it never collected.

So I picked what felt like the smallest piece of game state that is *actually state* — a
backpack with a count in it — and spent a month trying to make a 1.3B video model keep
one honestly. This repo is the result.

When I tried to do it the obvious ways, two things broke:

1. **Sparse actions aren't causal by default.** Movement and camera dominate every batch;
   `place` fires so rarely that it gets roughly 0.2% of the reconstruction gradient. Train
   on 62 hours with plain reconstruction and you get a model that places blocks whether or
   not the key is pressed (3/3 test windows). There's no event to account for if the event
   doesn't obey the button.
2. **Neural state read/write is unreliable at this scale.** I attached a held-item
   injector and watched its input gate converge to 0.002 — the first frame already reveals
   what you're holding, so gradients never bother reading the condition. Solaris runs into
   the same wall from the other side with 23× more data: place decrements its implicit
   inventory, but mining and pickup never credit it.

So the plan became: make the sparse action actually obey its bit (that's a training
problem), and build the bookkeeping itself out of things that cannot drift — plain code,
not weights.

## The inventory loop

<p align="center">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/architecture_v2.png" width="100%" alt="Architecture: DiT video model, three sensor questions, deterministic backpack with veto"/>
  <br/>
  <em>Actions drive a DiT video model (Wan2.1-1.3B base). Three questions are asked of its
  output: When? and What? are answered from the latents, Gone? from the decoded frames.
  A deterministic backpack does the arithmetic. Two paths feed back: the held item becomes
  conditioning, and an empty backpack blocks the action before it reaches the model.</em>
</p>

<p align="center">
  <img src="https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger/resolve/main/eval_assets/demo_ledger/sensors.png" width="100%" alt="Inside the three sensors, on a real mining event"/>
  <br/>
  <em>Inside the sensors, on a real mined block from the demos. <b>A</b>: a 2.5M-parameter
  CNN + temporal transformer reads the generated latents and scores every frame for mine /
  pickup / place events; the score spikes at the strike. <b>B</b>: the transition frame
  (where the block vanishes) is localized, the crosshair region is cropped in latent
  space, and a 16-class CNN names the item. <b>C</b>: crosshair region before vs after —
  the removed block lights up in the diff. The ledger applies plain rules on top: mine
  event + gone-verdict means +1 with provenance; a confirmed place means −1 of the held
  item. The probe is an event detector, not a success detector, which is exactly why the
  settlement check exists.</em>
</p>

## Teaching the model that one bit matters

The base is Matrix-Game-2's **bidirectional** Wan2.1-1.3B, fine-tuned on 62 h of VPT
gameplay. Generation is windowed image-to-video (50-step sampling, first frame anchored);
the demos chain windows by feeding one window's last frame in as the next window's
condition. A note on the word "causal": in this project it means *the action bit causes
the event*. It does not mean autoregressive architecture — see the limits section for
where that stands.

Reconstruction alone leaves `place` non-causal, for the gradient-share reason above. The
fix is a set of counterfactual losses built around one core move: **every window that
contains a real place event is run through the model twice** — same noised latents
$x_t$, same noise, same context, once with the place bit on and once with it zeroed.
Everything interesting lives in the difference between those two branches. From each
branch we take the model-owned belief $\hat{x}_0 = x_t - \sigma_t\, v_\theta$ and score
it with a frozen event probe, giving per-frame logits $z^{\text{on}}$ and
$z^{\text{off}}$.

The probe logits deserve one remark before the losses: an *absolute* probe score on
$\hat{x}_0$ is contaminated — $\hat{x}_0$ interpolates toward the ground-truth clean
latent, so the GT block leaks in and scores 0.75–0.97 "for free" (measured on a model
that provably couldn't place). Both branches share $x_t$, so the leak is identical on
both sides and **cancels in the difference** $z^{\text{on}} - z^{\text{off}}$. That
cancellation is why the whole design is differential.

The shipped objective is

$$\mathcal{L} \;=\; \mathcal{L}_{\text{rec}}
\;+\; \lambda_{\text{mar}}\,(\mathcal{L}_{\text{mar}} + \mathcal{L}_{\text{neg}})
\;+\; \lambda_{\text{on}}\,\mathcal{L}_{\text{on}}
\;+\; \lambda_{\text{ctr}}\,\mathcal{L}_{\text{ctr}}
\;+\; \lambda_{\text{out}}\,\mathcal{L}_{\text{out}}$$

with $\lambda_{\text{mar}}{=}0.25$, $\lambda_{\text{on}}{=}0.1$,
$\lambda_{\text{ctr}}{=}\lambda_{\text{out}}{=}0.5$. Term by term:

- $\mathcal{L}_{\text{rec}}$ — standard flow-matching reconstruction, with a spotlight:
  the crosshair region gets 30× weight for the five frames after an event. The spotlight
  is applied symmetrically to place events *and* to presses where nothing appeared, so it
  can't become a "press ⇒ block" shortcut; it just says *look here, this region decides*.
- $\mathcal{L}_{\text{mar}} = \mathrm{softplus}\big(m - (z^{\text{on}} - z^{\text{off}})\big)$
  on true place frames, $m{=}2$. The main course: on frames where a block really
  appeared, the ON world must out-score the OFF world by a margin. This is the term that
  makes the bit matter.
- $\mathcal{L}_{\text{neg}} = \mathrm{softplus}\big(z^{\text{on}} - z^{\text{off}}\big)$
  on *hard negatives* — frames where the button was pressed but no block appeared (aimed
  at the sky, out of range). The mirror image of the margin: pressing is not placing, and
  the model must not hallucinate success on invalid presses. Without this term the bit
  becomes a magic wand.
- $\mathcal{L}_{\text{on}}$ — a weak absolute BCE on $z^{\text{on}}$, weighted by
  $\sigma^2$ so it only bites where the model actually owns $\hat{x}_0$ (high noise) and
  the leak is weakest. Pure elicit pressure; the heavy lifting stays with the margin.
- $\mathcal{L}_{\text{ctr}}$ — before the press, the two branches must match pixel-wise.
  The ON and OFF worlds are genuinely identical until the button goes down, and this term
  says so. (An earlier version also tethered *post*-press frames to the ON branch; that
  quietly injected fake GT and had to go.)
- $\mathcal{L}_{\text{out}}$ — after the press, the ON−OFF difference is only allowed
  inside an allow-region: the crosshair box plus the lower-right arm area. Any difference
  outside it — global fog, color shifts, the model's favorite cheap tricks — is squashed.
  Together with the margin this forms a pincer: *there must be a difference, and it must
  live where the block and the swing live.*

Two terms are deliberately absent. An absolute suppression of the OFF branch
($z^{\text{off}} \to 0$ on event frames) sounds reasonable and was catastrophic: within
3k steps the model learned a smearing cheat that erased the probe's evidence instead of
suppressing the event, and the OFF branch degraded into hand-texture mush. It's still in
the code behind `W_POFF`, permanently set to 0, as a warning sign. A base-model preserve
term met a similar fate (it fights the GT block on post-place frames).

The schedule matters as much as the losses. Half of the event windows are forced into
the high-noise band $t \in [500, 1000]$, because that's where the model decides
*whether* the event happens — by low noise the die is cast. The probe terms are gated to
$t \le 800$, the probe's verified judging domain (AUC 0.984 at 800, chance below). Place
windows are oversampled to 50%, hard negatives to 10%. Action bits are binarized
(training used to encode `place` as its within-chunk duty cycle, 0.2–0.8, while
deployment sends 1.0 — a silent factor-of-two mismatch). And the new `place` column is
initialized from scratch rather than warm-started from `attack`: after 1000 steps of
warm-start training the two columns still had cosine similarity 1.0. Some basins you
don't escape.

At inference, classifier-free guidance over the action sharpens the event:
$v = v_{\text{off}} + w\,(v_{\text{on}} - v_{\text{off}})$, $w = 2.5$. Amusingly the axis
works in reverse too: $w = -2.5$ deletes blocks that were already there. The supervision
seems to have carved an actual semantic direction, not a texture trigger.

The result: two inputs differing by exactly one bit, and the block appears and persists
(paired probe score 0.994 vs 0.015). The model responds on the keypress and stays silent
without it in 12/12 held-out windows under the FIRST_ONLY protocol.

## Key results

| Claim | Number | Evidence |
|---|---|---|
| Counterfactual causality: respond on keypress, silent without | **12/12 windows** | `eval_assets/place_eval`, FIRST_ONLY protocol |
| Gold sample: inputs differ by exactly 1 bit; block appears and persists | probe **0.994 vs 0.015** | `firstON/OFF_15000.mp4` |
| Reconstruction-only baseline (earlier Fun-InP chassis, same VPT footage) | causality ≈ 0 | `cf_place_v3.log` |
| What-head: block identity at the transition frame (16 classes) | **80.4% val**; transfers to generated rollouts | `mine_id_head.pt` |
| Economy cycle (mine +1 → place −1 → veto) | all three acts pass | `economy_demo.mp4` |
| 4-act ledger lifecycle (place −1 / break +1 / place −1 / veto) | all four acts pass | `ledger_demo_4act_*.mp4` |

## Eleven measured failure modes

Getting one bit to matter took eleven diagnosed failures, each with an experiment behind
it. I think of this list as the actual contribution — a supervision-engineering playbook
for sparse events in small-data world models. The full list with measurements is in
[`RESULTS.md`](RESULTS.md); highlights:

1. **Reconstruction dilution** — sparse events get ~0.2% of the gradient; recon-only training moves the action column 1e-4 in 2000 steps.
2. **Teacher leakage** — absolute probe scores of 0.75–0.97 were all leakage; differential margins are immune.
3. **Warm-init contamination** — initializing `place` from the `attack` column leaves cosine 1.0 after 1000 steps.
4. **Behavioral echo** — keeping future action tracks lets the OFF branch get "scripted" into placing; the FIRST_ONLY protocol isolates it.
5. **Anchor-free counterfactuals** — absolute suppression of the OFF branch learns smearing-based cheating within 3k steps.
6. **CFG as a semantic axis** — w=+2.5 creates real blocks; w=−2.5 deletes even pre-existing ones.
7. **Direction asymmetry** — appearance (place) is learnable counterfactually; removal (mine) was never causally supervised anywhere in the lineage and only emerges in-distribution. It is distance-dependent (blocks only break at arm's length) and context-sensitive: a synthetic frozen-camera attack scores probe ~0.1, while replaying the real human action track in the same scene scores 0.67–0.88. This is why mine credits carry a pixel-settlement backstop.

## What this is, and isn't

Honest expectations, since this is a solo project and not a paper:

- **The demo clips are picked.** The place direction is the solid one — it was trained
  counterfactually and the 12/12 / 0.994-vs-0.015 numbers come from a fixed protocol, not
  from cherry-picking. The *mining* clips are another story: destruction was never causally
  supervised, so getting a clean "block breaks and stays broken" generation means rolling
  seeds and picking scenes. On a good scene 3 of 4 seeds pass the sensor gates; on most
  scenes none do. The clips you see are the survivors, and I'm telling you that up front.
- **There are no ablations.** Every loss term earned its place by fixing a measured
  failure during development, which is documented, but I never ran the controlled
  matrix — no time. If a term looks redundant to you, you might be right.
- **This is one game, one base model, 62 hours of data.** Think of it as a slightly
  over-the-top attempt to teach a pretrained video model *one new action* with a small
  dataset, plus the bookkeeping to make that action count for something. If you're
  attempting something similar, the failure-mode list above is the part most likely to
  save you time.

## Repository layout & reproduce

Weights, eval videos, the what-head dataset, and git bundles are all on
[HuggingFace](https://huggingface.co/yuzhewu207/mc-neurosymbolic-ledger). Run scripts
from the repo root.

| Path | What it does |
|---|---|
| `training/train_mg2.py` (via `fire_v5.sh`) | main training; margin / t-band / L_out behind env switches |
| `training/train_mine_id.py` | what-head: 16-class mine identity from transition-frame latent crops |
| `ingest/` | VPT → latents + event labels (`vpt6xx_ingest.py`), QC (`census_jsonl2_qc.py`) |
| `evals/` | counterfactual eval (`cf_stage1.py`, `batch_place_eval.py`, `probe_t_sweep.py`), v3 recon baseline (`cf_place_v3.py` + legacy deps) |
| `demos/` | the inventory demos: `demo_economy.py`, `demo_ledger_minimal.py`, `demo_attack.py` |
| `RESULTS.md` | full results, failure-mode chain, mechanism findings |

The active pipeline runs entirely on Matrix-Game-2's code and weights (the scripts import
from a local Matrix-Game-2 checkout). The `wan/`, `model/`, `pipeline/`, `trainer/`,
`utils/` packages in this repo are legacy — they only serve the v3-baseline scripts in
`evals/`. Hardware: 4× RTX Pro 6000.

Trivia: the project started as a V Rising world model and pivoted to Minecraft --
the development repo is still called `vrisingwm`.

## Known limits

- Mine-direction settlement is pixel-based. At deployment this costs nothing (frames are
  decoded for display anyway), but it means accounting over *undecoded imagined futures*
  only works for the place direction today. The fix path is measured and open: use the
  pixel settler to self-label generated rollouts, then distill a disappearance head back
  into latent space.
- The what-head confuses gray stone-family blocks (stone 53%); logs, grass and sand sit
  at 82–95%.
- The shipped generator is bidirectional within each window; interactivity comes from
  chaining windows, not real-time autoregression. An AR-causalized variant (ForgeWM
  stage-1 recipe, 1k steps) transfers dense actions but not the sparse `place` event yet.
  Making counterfactual supervision survive causalization is open work.
- Single game, single base model, 62 h of data.

## Acknowledgements

Built on [Matrix-Game-2](https://github.com/SkyworkAI/Matrix-Game) (base model and
inference code), [Wan2.1](https://github.com/Wan-Video/Wan2.1) (architecture),
[VPT](https://github.com/openai/Video-Pre-Training) (data), and
[Causal Forcing](https://github.com/thu-ml/CausalForcing) (recipe for the AR track).
Thanks for their wonderful work.

## Citation

```bibtex
@misc{wu2026inventory,
  title  = {A World Model That Maintains Its Inventory: Counterfactual Supervision for
            Sparse Interaction Actions and a Neuro-Symbolic Ledger},
  author = {Wu, Yuzhe},
  year   = {2026},
  note   = {https://github.com/wuyuzhe-svg/mc-neurosymbolic-ledger}
}
```
