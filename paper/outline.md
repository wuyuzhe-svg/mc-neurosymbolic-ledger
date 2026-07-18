# 【已存档 2026-07-18】AAAI 全文骨架(决定不投;证据地图仍可用作 README/文档索引)

每节标注:内容 → 证据文件(全部已在 HF repo `yuzhewu207/mc-neurosymbolic-ledger`)。

## 1 Introduction(1 页)

- 钩子:世界模型的动作分层——密集动作(WASD/相机)人人都会,稀疏交互
  (place/mine)决定"世界"是否有经济;现有系统(MG2/GameCraft/Oasis)
  用 23× 数据规模绕过,我们问:小数据下能否用监督工程直接解决。
- 图 1:经济循环 demo 三联(挖+1 → 放−1 → 拦截),配 what-head 识别标注。
  素材:eval_assets/demo_ledger/economy_demo.mp4 / _strip.png
- 贡献三条(见 abstract.md)。

## 2 Related Work(0.75 页)

- 交互世界模型三路线表(PROJECT_MASTER §2:双向蒸馏 A / 因果直学 B / 无教师 C)。
- 神经-符号状态:Solaris 隐式库存(不对称:−1 有 +1 无,我们的受控实验,
  memory: solaris-inventory-asymmetric)作为"纯神经写接口天花板"的对照锚点。
- 反事实/事件监督:Causal Forcing(底盘)、GameFactory(相机先例)。

## 3 Problem & Failure Analysis(1 页)——方法论主战场

- 3.1 重建稀释定量:place 梯度份额 ~0.2%,纯 recon 2000 步动作列位移 1e-4;
  v3 基线 3/3 窗"不按也放"。证据:cf_place_v3.log
- 3.2 失败模式链表格(11 条,RESULTS.md §失败模式链逐条搬运,每条一行:
  现象 / 测量 / 修法)。这是审稿人记住我们的地方。

## 4 Method(1.5 页)

- 4.1 底座与数据:Wan2.1-1.3B (MG2 init) + Causal Forcing chunkwise;
  62h VPT,哈希互斥三路划分(clean-split-leakage-fix)。
- 4.2 事件级反事实监督:差分 margin L = max(0, m − (z_on − z_off));
  t-band σ∈[0.5,1.0](高噪=事件决定区,探针裁判域实测到 σ=0.8);
  二值化动作编码;de-warm 重置;为什么绝对压制(Lpoff)危险(涂抹作弊)。
- 4.3 探针体系:事件探针(when)+ what-head(16 类,跳变帧准星区 crop);
  差分 margin 免疫 teacher 泄漏的论证。
- 4.4 账本闭环:探针 fire → 确定性记账(mine +1 / place −1);账本 → 
  动作位掐断(拦截);held 注入(账本解析)。图:系统框图。

## 5 Experiments(2 页)

- 5.1 主表(RESULTS.md 主结果表搬运):
  12/12 FIRST_ONLY;金样本 0.994 vs 0.015;CFG w=2.5 持久放置 3/12 严格;
  recon 基线 ≈0;经济循环三幕;四幕账本 demo;AR 迁移(稠密 ✓ 稀疏 ✗)。
- 5.2 消融:t-band 扫描(probe_t_sweep)、margin vs Lpoff、二值化 vs 占比、
  de-warm vs warm、FIRST_ONLY vs 保留剧本(行为回声)。
- 5.3 机制实验:CFG 剂量-反应轴(w=+2.5 生块 / w=−2.5 负放置);
  held 门权重 0.002(上下文冗余锁死);破坏距离消融(时长无关、距离依赖)。
- 5.4 what-head:16 类 val 80.4%(段级划分),生成态 sand@1.0,
  跳变帧=入账时刻的语义同构。
- 评测协议:人评终审制(探针=事件检测器≠成功检测器,失败模式 5)。

## 6 Discussion & Limitations(0.5 页)

- 方向不对称:出现(place)可反事实监督,移除(mine)只在分布内涌现;
  +1 记账靠像素级消失结算而非神经识别 → 诚实声明。
- 灰色石系互混(stone 53%);单游戏单底座;62h 上限。
- 账本必要性论证(神经写接口小数据不可靠 → 符号侧给保证)放这或 §4.4。

## 7 Conclusion(0.25 页)

## 写作日程(倒排)

- 7/18–19:摘要定稿 + §3/§4 初稿(方法论是卖点,先写)
- 7/20:§5 表格与图全部落位(素材已齐,不需要新实验)
- 7/21:**摘要提交**;§1/§2
- 7/22–24:如审稿人视角预检(preflight-adversarial-audit)发现漏洞,
  用这窗口补消融;否则全文打磨
- 7/26:全文冻结自查;7/28 提交
