# DGX Spark 无交换环网与 TP4 方案全网调研报告

> 调研日期：2026-10-07 · 方法：5 路并行检索（GitHub repo search 初筛 ~150 仓库、官方文档、NVIDIA 论坛、Reddit/HN、博客媒体、中文社区）+ 当日二次核实（GitHub API 元数据逐仓验证 + 本机 spark-bf94 只读核查）。
> 关联笔记：《GLM-TP4环网部署方案总结.md》《本地AI部署总览.md》；上篇：jnardiello 方案对比（会话记录）。
> 标注约定：〔核实〕= 本次直查证实；〔摘录〕= 来自仓库原文/agent 摘录但未逐字复核；〔存疑〕= 未能确认。

---

## 0. 二十条头条结论

1. **官方支持上限是 4 台，且 4 台官方要求交换机**。官方时间线：2025-10 首发只支持 2 台直连 → GTC 2026 官宣 4 台 → 2026-06 Cluster Assistant 发布（直连最多 3 台、4 台必须经交换机）+ NCCL 2.30u1 官方支持 3 台环形拓扑。**4 台无交换环始终是社区自留地**，但目前已被 10+ 个独立项目生产化跑通。
2. **knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 自 2026-09-29 起内置 `TRANSPORT=switchless`**（社区 PR #1，作者 othexmr，rebase 到 3ab5ca0）。〔核实〕我们 10-06 的部署**实际已经在用这套机制**：`.env.switchless` 里就是 `TRANSPORT=switchless` + 自建 NCCL 的 SHA256 + `IB_HCA` 四 PF。作者自己的机群在交换机后、"cannot boot it"——**我们的环就是这套代码唯一的硬件资格认证，值得回报上游**。
3. 〔核实〕**我们的"双平面"物理上是 4 根线 × 每线 2 rail**。spark-bf94 实查：每台 4 个 RDMA function（rocep1s0f0/f1 + roceP2p1s0f0/f1），每个 function 1 个 port，每台只有 2 个物理 QSFP 口。笔记里"每对邻居 2 根线，共 8 根"应修正为"4 根线、每线 2 条 Socket-Direct rail"——与 SparkRing dual-rail/双域补丁完全同构。
4. 生态地基三件套：**eugr/spark-vllm-docker**（2363★，MIT，社区 vLLM 镜像事实标准，"direct/QSFP switch/3-node mesh/closed rings of four or more" 全支持）〔核实〕；**spark-arena/sparkrun**（534★，Apache-2.0，无 Slurm/K8s 的多机编排，ring/mesh 自动注入 NCCL 变量）〔核实〕；**FujitsuPolycom/sparkring**（88★，Apache-2.0，一体化栈 + 一键 installer + 多模型 profile + 性能记录）〔核实〕。
5. **TensorFold（ashhart/TensorFold，1077★，Apache-2.0）成为 vLLM/SGLang 之外的第三引擎路线**〔核实〕：Metal/CUDA/Vulkan 推理引擎，双机实测 1.8× vLLM decode（jayleaton），正在被 Keenni/EdwardHodl/drowzeys 等推向 4 机 TP4 环。
6. **DeepSeek-V4.1-Flash 是环上生态最厚的模型家族**：LuZ（28★，SGLang TP4 环生产配方，PR 峰值 5,823 t/s）、knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4（48★，docs/switchless-ring.md 完整环网 runbook）、SparkRing 官方 profile（57/135/201/275 @1/4/8/16 用户）、MiaAI-Lab 上游。其 decode allreduce 仅 61 KB/层（LL 协议，延迟主导），**环网友好度全场第一**。
7. **RoCEnante one-shot allreduce 是 decode 提速的公共武器**：10 KiB allreduce 16.9 µs vs NCCL ring 90.6–92.4 µs（christopherowen 4 机环实测）。但它要求 all-peer 直达——4 机环上对角路径靠 **ConnectX-7 硬件 hairpin**（EtherType 改写 0x0800↔0x88b5、中间节点 ASIC 硬件转发）实现。b12x 仓库已归档、迁往 FlashInfer（merge PR #5767）。
8. 〔核实〕GB10 **无 GPUDirect RDMA**（官方 FAQ：UMA 下 peermem/dma-buf/GDRCopy 均不可用）——所有方案都是 CPU posting + host pinned bounce buffer。这是各家做 one-shot/自研传输的根因；`dgx-spark-networking` 已把 GPU-initiated networking（DOCA GPUNetIO）列入路线图。
9. **线缆层环 = 交换机**：Petronella 实测单根直连 RDMA 196 Gb/s，经 CRS812 交换机同样 196 Gb/s。差距在非邻居流量的跳数/共享带宽与长稳性，不在带宽。alexellis 的供应链论点："switchless ring removes the switch, and its lead time, from the critical path"（MikroTik CRS804/812 长期缺货是社区选环网的最大动因）。
10. **运维红线：线缆静默劣化**。SparkRing 记录：一根线双向带宽 213→26 Gb/s 而 RoCE 计数器全 0，单向测试测不出；TP2 prefill -34~37%。巡检必须 `ib_write_bw -b -x 3` 双向逐 function，健康阈值 ≥190 Gb/s。
11. NCCL on GB10 已知坑清单（见 §6.6）：2.30.4+ NVLS 在 MoE 加载静默挂死（`NCCL_NVLS_ENABLE=0`）、`NCCL_IB_MERGE_NICS=0` 必须 pin、HCA 名大小写敏感（小写 P 静默回落单 rail）、GID index 3 依赖 IPv6 link-local（禁 IPv6 会漂移）、**MTU 1500 让 decode 慢 ~2.7× 且无报错**。
12. **上下文军备竞赛到 1M**：yunwei37（993,435-token needle @1,048,576 窗口）、METIQ（1,048,283 验证）、sethforprivacy（1,007,197）、pd-bridge 服务窗口 2M。KV 手段：FP8/NVFP4 KV、Engram NVMe 化、sparkcache 磁盘持久缓存、NVMe offload tier、UVM 64KiB 页表补丁。
13. **8 机路线兴起**：TP8 + 自研 ibverbs one-shot allreduce（EternaPeptix：48KB AR 150–190 µs→34 µs；roofline：8×Spark 无投机 ≤53 tok/s）；8 机 DeepSeek EP 化"removes most of the all-reduce wait"。NCCL 通道数 "2 channels beat 16 by 33–48%"（decode 消息尺寸）。
14. **dense 模型的教训**：Qwen3.6-27B TP4 单用户 2.62×，但 C=64 下 TP2→TP4 效率仅 59%，"TP=4 close to the useful limit"；结论 "Latency → TP，throughput → replicas"。Llama-405B 被社区否定；gpt-oss 吞吐场景 PP=2 比 TP=2 快 2.2×（PP 通信模式与环拓扑天然同构）。
15. **量化路线**：全 NVFP4 被 jnardiello 保真度审计证伪（ppl +7.1%、>2K token +12.1%）；存活的是 knapcio 的 NVFP4-experts+lossless8 混合、jnardiello 的 FP8+INT8、EXL3（MiaAI）、Int4-Int8Mix（full 753B）。TensorFold 系走 EXL3/4bpw。
16. 成本量级：4 机交换机方案 ≈ **$1,100–1,500**（CRS804 列表 $1,295 街价 ~$1,100，缺货是主要矛盾）；SN3700 200G 翻新 $21.8k 是另一个世界；200G DAC 第三方 $45–54/根、Amphenol 认证线零售 $159/根；环网只需 4 根线 ≈ $200–640。
17. **DFlash2 drafter 是 CC BY-NC-ND（非商用）**，是全生态公共约束；商用替代：canada-quant/GLM-5.3-Flash-DFlash2-G（Apache-2.0）或 checkpoint 自带 MTP head。
18. **PD 分离/异构是新增长点**：pd-bridge（150★，Spark TP2 prefill → Mac Studio decode，700K prompt 11 分钟端到端）、tpurtell/hughmadden 的 Spark+RTX PRO 6000 AFD（attention 放 5090、experts 放 4×Spark）、macncrash/sparkwire 的 RDMA over USB4 原型。
19. 学术侧背书：arXiv 2605.00254（switchless mesh 比 scale-up 性价比高 20.6–56.2%，NVIDIA Research 提供数据验证）；arXiv 2603.28239（SCIN 在网计算对 ring allreduce 小消息 8.7×——反证普通 ring 的通信上限）。
20. 中文社区深度内容少：超擎数智有 2/4 机 NCCL 基准文、腾讯云转载教程、B站双机/四机实测视频，但**没有 alexellis 级别的完整中文环网复现教程**——我们的笔记在中文圈就是稀缺内容。

---

## 1. 官方立场与硬件事实

### 1.1 NVIDIA 官方多机路线（时间线）
- 2025-10 首发：2 台直连（"daisy-chaining"），405B（FP4+sparsity）上限口径。
- GTC 2026（3 月）：官宣 4 台 "desktop data center"（https://blogs.nvidia.com/blog/gtc-2026-news）。
- 2026-06 软件更新：Cluster Assistant 发布（**直连 2–3 台、交换机 4 台**，不得混用，每链路一根线）+ NCCL 2.30u1 官方支持 3 台环形（https://forums.developer.nvidia.com/t/dgx-spark-software-updates-june-2026-release/371965）。
- 4 台以上：无路线图。官方答复"加第二个 CX-7 口做级联"无计划；8/16 机全靠 MikroTik + 社区经验。
- 官方 playbook：`connect-two-sparks`（"Full bandwidth can be achieved with just one QSFP cable"）、`connect-three-sparks`（官方环形！p0→p1 交叉）、`multi-sparks-through-switch`、`nccl`（源码编译 Blackwell NCCL）。https://github.com/NVIDIA/dgx-spark-playbooks

### 1.2 GB10 网络硬件（理解一切组网的前提）〔核实，多源一致〕
- 一颗 ConnectX-7 ASIC、两个外部 QSFP 口；NIC 以**两条独立 PCIe Gen5 x4** 接入 SoC（multi-host 聚合才有 200Gb/s——GB10 SoC 每设备只给 x4）。
- 每物理口 2 个 function（分属 PCIe domain `0000`/`0002`）→ 每台 4 个 RDMA 设备。**本次在本机核实**：`ibdev2netdev` = rocep1s0f0/f1 + roceP2p1s0f0/f1，各 1 port。
- 单 function 单向 ~109 Gb/s（Gen5 x4 极限），双 function 双向 ~196–213 Gb/s（SparkRing `ib_write_bw -b -x 3` 实测；geerlingguy：function 对齐错误时两条流挤一条链路只有 92–95 Gbps）。
- **GPUDirect RDMA 不支持**（UMA）：官方建议 cudaHostAlloc + ib_reg_mr 的 host-pinned bounce buffer。
- 端口约定：p0 = 靠近 10GbE RJ45 的口；环接法"每台 p0 → 下台 p1"。
- 认证线缆：Amphenol NJAAKK-N911 / NJAAKK0006、Luxshare LMTQF022-SD-R。

---

## 2. 生态全景地图（按类别）

### (a) 4 机无交换环网方案（我们的同类，14+ 个）
| 项目 | ★ | 模型/引擎 | 要点 |
|---|---|---|---|
| alexellis/glm-5.3-flash-4x-dgx-spark-switchless | 104 | GLM-5.3-Flash NVFP4 + DFlash2 / vLLM | 第一个 4 机环配方；switchless-nccl v0.0.1；Rigmark code 71.3/prose 31.2 |
| jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless | 30 | GLM-5.3-Flash FP8+INT8 / vLLM | E36 基线 code decode 65.55；SparkCache 磁盘重放 64K 40,002 tok/s |
| ntxf31415/glm-5.3-flash-nvfp4-4x-…-switchless | 3 | GLM NVFP4 | 单流峰 94.7/均值 75.9；400K 冷 prefill 2,097 tok/s〔摘录〕 |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-…-RiNGSiDE | 31 | GLM NVFP4 | TP2/TP4 双 profile；**knapcio PR #1 的作者** |
| luxingcom/LuZ-0.1.7-DeepSeek-v4.1-…-TP4-Ring | 28 | DSV4.1 MXFP4 / SGLang | 生产级环配方：PR 峰值 5,823 t/s、C16 聚合 572.6 t/s、C1 code 89.19〔摘录〕 |
| knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 | 48 | DSV4.1 / SGLang | docs/switchless-ring.md 环网 runbook；prose c1 89.7/code 131.9（switched 口径）|
| yunwei37/dgx-spark-4-ring-no-switch | 15 | DSV4.1/GLM | 1M 窗口 993K needle；50.14 tok/s C1；记录文化最佳〔核实〕|
| FujitsuPolycom/sparkring | 88 | GLM/Qwen/MiMo/DSV/Swift 全 profile | 一键 installer；双后端 RoCEnante+NCCL〔核实〕|
| christopherowen/dgx-spark-networking | 24 | 传输层库 | 10-04 新出；oneshot+NCCL 双后端按 collective 选路〔核实〕|
| sethforprivacy/mimo-v26-flash-spark-ring-recipe | 0 | MiMo-V2.6-Flash | 1M needle 通过；RigMark 75.8/121.7/128.2 |
| METIQ-Solutions/deepseek-v4.1-gx10-switchless | 0 | DSV4.1 | 4×GX10 环 1M 参考实现；1M TTFT 949s |
| nero-/deepseek-v41-flash-gb10-ring | 0 | DSV4.1 | **2×Spark+2×GX10 混合环实证**（跨品牌组环可行）|
| run1999、ChrisLou-bioinfo、Richard0571、masquerator(TP3)、GroveMinting、luxingcom/LuZ0.4.5 等 | 0–2 | DSV4.x | 各自环上复现/教程（多为中文）|
| ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe | 5 | DSV4.1 / SGLang | 1M ctx；维护 **跨项目对照表**（results/2026-09-28-cross-project…md）〔存疑：直读超时，数字来自 agent 摘录〕|

### (b) 多机带交换机/通用多机
MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks（218★，事实标准上游，TP3 三角+TP4 双 profile）、tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark（94★，8机 EXL3 记录）、rhys101 8机（TP8/EP4+RoCEnante，交换 fabric）、sumsliu 8机（TP8+EP8，"EP removes most of the all-reduce wait"）、ciprianveg/gb10-glm-5.2（38★，8×GB10 TP8 GLM-5.2 vision）、EternaPeptix/mimo-v2.6-pro-dgx-spark-tp8（8×Spark ~1T，自研 AR 34µs）、GumbiiDigital 8机工程、mark-ramsey-ri/vllm-dgx-spark（131★，1-to-N 41 模型 preset）、tonyd2wild/GLM-5.2-QuantTrio（76★）、SBE247Emperor 8×GLM-5.3 FP8、chishiki37/dgx-spark-fabric（MikroTik 交换 2 机，"+26-33% decode"）、localaiguyy（Cisco Nexus 3232C 参考，〔存疑：可能是 Apache Spark 噪音〕）。

### (c) 双机直连（stock NCCL 即可）
eugr/spark-vllm-docker（2363★地基）、MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-TensorFold（281★）、tonyd2wild TP2（218★，DFlash2 首上 GB10，acceptance 74.1%）、raullenchai/twinspark（176★）、chadhurley25075-png/pd-bridge（150★，PD 分离）、myllmann...（Qwen3.8 TP2 117★）、jayleaton/glm53-tensorfold-spark（109★）、Anemll/dspark-vllm-gx10（81★）、MiaAI DSV4.1 EXL3 2x（257★）等。

### (d) 引擎/单机多机化技术
ashhart/TensorFold（1077★〔核实〕）、local-inference-lab/b12x（272★，**已归档迁 FlashInfer**）、AEON-7 vllm-ultimate（144★ sm_121a 源码构建）、timothystewart6/vllm-gb10（53★ upstream-first）、christopherowen/dgx-spark-memory-saver（UVM 64KiB 补丁）、Avarok-Cybersecurity/atlas（705★ Rust 引擎 + sparkrun recipe 表）。

### (e) 工具/监控/基准
spark-arena/sparkrun（534★〔核实〕）+ styles01/sparkrun-recipes（82★）、MiaAI-Lab/sparkDash（548★，knapcio 系标准 harness）、alexellis/rigmark（Show HN 10-06）、joeynyc/spark-doctor（160★ 诊断 CLI）、niklasfrick/spark-dashboard（130★）、metaspartan/sparktop（CX-7 fabric 遥测）、RodriMora/grafana-dashboard（40★）、bidual/awesome-dgx-spark（76★）、mberta/link-test、digchick/200g-link-fix、ajclark/nccl-plugin-low-power、kajikuni/admission-gate、joesinvestments/gx10-bench-optimizer、NVIDIA/dgx-spark-playbooks（官方）。

---

## 3. 环网方案详细对比（重点）

### 3.1 关键数字总表（口径各异，禁止直接横比）

| 项目 | 拓扑/传输 | 模型/量化 | decode tok/s | prefill | 口径 |
|---|---|---|---|---|---|
| SparkRing glm53 tp4〔核实〕 | 4 线环+hairpin；RoCEnante+NCCL2.32.3 | GLM NVFP4 | 62.1/133/200/231 @1/4/8/16 | 64K 3,500 | 16K ctx、thinking on·max、temp1.0、"not serving-qualified" |
| SparkRing dsv41 tp4〔核实〕 | 同上 | DSV4.1 FP8/MXFP4 | 57.0/135/201/275 | 64K 4,396 | 同上 |
| SparkRing qwen38 tp4〔核实〕 | 同上 | Qwen3.8 NVFP4 | 64.0/173/239/328 | 64K 4,424 | 同上 |
| SparkRing mimo tp4〔核实〕 | 同上 | MiMo-V2.6 | 62.8/124/181/317 | 64K 4,094 | 同上 |
| alexellis | 4 线环；switchless-nccl v0.0.1 | GLM NVFP4+DFlash2 | code 71.3/prose 31.2；C1 43.0/C2 72.5/C4 110.8 | 64K cold 1,965；replay 36,075 | Rigmark v1.0.0、reasoning=low、n=5 中位 |
| jnardiello E36 | 4 线环单平面 4 DAC | GLM FP8+INT8 | code decode 65.55/prose 34.64；C4 107.60 | 64K cold 2,789；replay 40,002 | Rigmark upstream、n=2×108 |
| knapcio（我们自己的栈） | 4 机环；SparkRing dual-domain NCCL | GLM NVFP4-exp+lossless8 | prose c1 71.3/code 115.1（sparkDash）；qeval 73-75/75 | 8k-128k 冷 3,064-3,470 | sparkDash/仓库 bench、2550MHz |
| LuZ 0.1.7〔摘录〕 | 4 线环；ring-only NCCL+libncclpin | DSV4.1 MXFP4 / SGLang | C1 code 89.19；DE 聚合峰 572.6 @C16 | PR 峰 5,823（65536×C2）；512K 单流 5,042 | SD-1、600K ctx、9.6M KV、16 并发 |
| yunwei37〔摘录〕 | 方环 0↔1↔2↔3↔0；NCCL 2.29.7+mesh | DSV4.1 DSpark k=5 | 50.14 C1 mean/68.06 code；144.71 @8 req | — | 1M 窗口 993K needle；87-92% prefix hit |
| ntxf31415〔摘录〕 | 4 线环 | GLM NVFP4 | 峰 94.7/均值 75.9 | 400K 冷 2,097 | sparkDash 系 |
| ZackO2o〔摘录〕 | 环+sparkring NCCL | DSV4.1 SGLang | C1 104.10；C8 聚合 245.98 | — | 1M ctx |
| sethforprivacy〔摘录〕 | 4 线环；b12x RoCEnante ring-only | MiMo-V2.6-Flash | RigMark prose/code/structured 75.8/121.7/128.2 | 冷 64K/128K/250K = 3,555/3,025/2,379 | 1M needle 1,007,197 |
| METIQ〔摘录〕 | 4×GX10 环 | DSV4.1 | 128K ctx decode 35.24 | 128K 有效 1,910 | 1M TTFT 949.6s |
| MiaAI TP4〔摘录〕 | 交换机+RoCEnante（对照） | DSV4.1 SGLang | 上游 45.4/72.9/103.1/114.1/134.2；新版 87.7/120.4/163.6/237.6/342.7 | 冷 64k/128k 5,925/5,797 | 两版本口径并存 |
| joesinvestments 论坛〔摘录〕 | 4 机（Fleet=2×TP2 对照） | DSV4F 0731 | Fleet 75.7 vs TP4 55.1 @8192 | Fleet 4,023 vs TP4 2,544 | 长新 prompt 并发突发 Fleet 更好 |

### 3.2 几个结构性观察
- **环上 decode 并不输交换机组网**：SparkRing 环 profile 数字（GLM 62-231、DSV 57-275）与 MiaAI switched 数字同量级——因为 decode 小集合通信走 one-shot/hairpin 直发，大集合通信用与物理环重合的 NCCL ring 算法。真正吃亏的是"无 RoCEnante 的纯 NCCL decode"（knapcio switchless.md 明示 "expect decode to be slower"）与 prefill 的单链路 bisection。
- **prefill 冷态各家高度收敛在 2,000–3,500 tok/s**（64K 口径），SparkRing 3,500 与我们 3,064-3,470 同档；重放/缓存才是差异点（jnardiello 40K、alexellis 36K replay——SparkCache/前缀缓存红利）。
- **谁在用谁的传输**：eugr 镜像是地基；b12x/RoCEnante 是 decode 提速公共件（已迁 FlashInfer）；josephdrose 补丁经 SparkRing（Apache-2.0）和 alexellis（switchless-nccl）两条线再分发；othexmr 四 PF profile 进了 knapcio 主线。
- **混合硬件可行**：nero- 用 2×Spark + 2×ASUS GX10 混组环成功〔摘录〕；但注意论坛报告 GX10 第二 CX-7 槽位只配了 Gen5 x2（比 Spark FE 半带宽）〔摘录〕。

---

## 4. 其他模型家族的环网适配性排名

| 排名 | 模型 | 环网友好度 | 依据 |
|---|---|---|---|
| 1 | DeepSeek-V4.1-Flash | 已完成环网化 | 三个独立生产实现（LuZ/knapcio-DSV41/SparkRing）；decode AR 61KB/层；要动的只有 patched NCCL+本地权重+管理网分离 |
| 2 | Qwen3.8-Flash-Next | 零改造 | SparkRing 现成 TP2/TP4 环 profile；EP 切 MoE；6B active 小激活 |
| 3 | MiMo-V2.6-Flash | 零改造 | SparkRing 官方 profile；1M needle 通过 |
| 4 | Swift-1.5-Qwen3.8 | 零改造 | SparkRing TP4 profile 64.7/161/232/340〔核实〕 |
| 5 | gpt-oss-120b/20b | 低（PP 路线） | 单机放得下；PP=2 2.2× TP=2（吞吐）；PP 通信=链形，与环同构；无现成 4 机环方案 |
| 6 | Qwen3.5-397B-A17B | 低 | sparkrun 官方支持 3 机无交换 mesh（PP=3）；扩 4 机未验证 |
| 7 | MiniMax-M3 | 中 | TP3 三角无交换全互联（每对直连）；上 4 机环需改 TP4 sharding 补丁 |
| 8 | dense 系（Llama/Qwen dense） | 中-高 | 每层 AR 无专家卸载；TP4 效率差；只推荐 PP 或副本；Llama-405B 被否定 |
| 9 | Kimi K2.5/K2.6 | 高风险 | INT4 在 SM121 被 Marlin MoE 内核 bug 阻塞（vllm#40357 closed not planned）；≥8 台起 |
| 10 | 图像/视频（Flux/Wan） | 不适用 | 无 Spark 多机先例；xDiT 理论可移植但大张量 allgather 在环上逐跳放大 |

DeepSeek 网络压力量化（原文数字）：decode allreduce 61 KB（LL）；NCCL 默认连接缓冲在 GB10 上 **4.7 GiB/节点** → `NCCL_BUFFSIZE=1048576`+`LL128_BUFFSIZE=262144`+`NCCL_PROTO=^LL128`+`MAX_NCHANNELS=8` 后降到 **0.14 GB**；"2 channels beat 16 by 33–48%"。

---

## 5. TP4/多机方案与"环网化潜力"评估

| 方案 | 当前拓扑 | 环网化难度 | 说明 |
|---|---|---|---|
| **knapcio GLM-5.3-Flash** | 交换双 rail + RoCEnante；**已内置 TRANSPORT=switchless** | **零**（我们已在用）〔核实〕 | 上游 rebase（3ab5ca0）作者未硬件认证；我们的部署就是认证 |
| knapcio DeepSeek-V4.1 | 交换 + RoCEnante；docs/switchless-ring.md | 零-低 | 已有完整环网 runbook + PR#8（DSV41_ROCE_RING） |
| MiaAI-Lab DSV4.1 | 交换/三角 | 低 | 下游 knapcio-DSV41 已环化 |
| tonyd2wild GLM TP2 / TP4-1M-KV | 双机直连 / 4 机单 rail | 低 | TP4 仓现默认跑 knapcio 栈；"Arista 100G 复现 decode 等速，not fabric-bound" |
| alexellis | 已是环 | — | 另有 switchless-nccl 独立构件 |
| Wpnx330 GLM FP8 | 4 机（fabric 未说明） | 中-高 | launcher 无预检无 switchless 概念，NCCL env 用户自填 |
| MiaAI EXL3-TensorFold | 双机 | 中 | TensorFold 引擎多机化进行中（Keenni 已上 4 机环） |
| 8 机系（rhys101/sumsliu/EternaPeptix/ciprianveg） | 交换 fabric | 高（需交换机） | TP8 自研 AR 34µs 是方向，环拓扑上 ring-only 也可行但无先例 |
| joesinvestments DSV4F TP4 | 4 机 | 低 | Fleet(2×TP2) vs TP4 的对照数据有参考价值 |
| chishiki37 Kimi K3 IQ1_S 3 机 | RDMA layer-split | 中 | llama.cpp RPC 路线，非 NCCL |

---

## 6. 传输层构件深解

### 6.1 josephdrose/nccl-spark-switchless（首创，无 license）
- NCCL 2.30.7 补丁：`relay.cc` 2-hop store-and-forward relay（对角线软件中转）+ `NCCL_SKIP_TREE_CONNECT`（跳树连接）双模式。对角经邻居中转全走原生 RoCE，多通道修过 4 个 bug（QP 队列深度谎报、阻塞转发死锁、deliver-before-register 竞态、send-slab 覆写）。
- 实测：ring 模式 MiniMax-M3 ~24 tok/s（vs 10GbE 回退 3.3）；树+relay 16.7 tok/s。
- 后继：joe-spark-patches（4 机 vLLM out-of-tree 补丁 + 批处理化 DSpark proposer）。

### 6.2 SparkRing（FujitsuPolycom，Apache-2.0，最完整）
- **SIRCL**：自研 4-rank native 传输，CUDA graph 内 device-command 直通；480 KiB AR p50 174 µs（NCCL 回退 ~800 µs 底噪）；只服务退役 profile 和 research 配置。
- **RoCEnante**（源自 b12x#295，Luke Alonso/Jason Cook）：每 rank 6 条持久 origin QP；对角 = CX7 硬件 hairpin（源端 steering 规则改 EtherType 0x0800→0x88b5、中间 ASIC 还原转发）；Q24 直达+对角阈值；发布后失败 fail-stop（exit 70）。b12x 已归档迁 FlashInfer。
- **patched NCCL 补丁族**（2.30.7 线）：switchless-cycle、skip-tree-pat、advertise-all-listener-gids（解决"NCCL 只发 2 个 listener GID，多过的设备零字节"）、dual-pci-domain（累计包，4-GID+PCI-root 保持）。实测 84 MiB AR **6.564 ms vs 13.354 ms（2×）**，serving prefill +5-7%，decode 无收益，4/8 channel 无差。
- **mHC prefill 分片**：reduce-scatter 部分和 → 每 rank 只算自己 2,048 行 → all-gather；+4.59–5.26% prefill。
- **SparkCache**：vLLM 持久 rank-local NVMe 前缀缓存（hash→最短全 rank 可用前缀→恢复；copy-on-write；流量全本地）。
- 安装器已升 NCCL **2.32.3**（kraken 镜像）。

### 6.3 alexellis/switchless-nccl（Apache-2.0）
NCCL 2.30.7 + SparkRing 两个补丁 + OpenFaaS 加固第三补丁；`NCCL_SWITCHLESS_RING_ONLY=1`（别名 SKIP_TREE_CONNECT）；SHA256 双 pin；公开 gotchas：MTU 9000 必须（否则 decode 慢 2.7×）、MERGE_NICS=0 必须 pin、GID 槽位漂移、fabric 配置 runtime-only（重启重跑 fabric-setup.sh、docker churn 冲掉 DOCKER-USER 规则）。

### 6.4 othexmr 四-PF profile → knapcio PR #1
四 device/四 listener；**硬编码 10.100.224.0/24–227.0/24**（different addressing needs separately validated build）。〔核实〕我们的 lib 无此字面量——我们跑的是 SparkRing dual-domain 系（参数化寻址 10.200/16），两套并存不冲突。

### 6.5 christopherowen/dgx-spark-networking（10-04 新，Apache-2.0）
双后端按 collective 选路：小 AR → `sparknet.oneshot`（10 KiB 16.9 µs / 480 KiB 94 µs / 2 MiB 307.7 µs），大 → NCCL ring（2 MiB 359 µs）。**均衡 channel 策略让 2 MiB 320→232 µs、四接口各承载 ~25%**。路线图：DOCA GPUNetIO GPU-initiated networking。自带 tp4-ring/tp3-triangle/switched profile。

### 6.6 NCCL on GB10 已知问题清单〔摘录〕
| Issue | 要点 |
|---|---|
| #2167 | 2.30.4+ NVLS 在 8×Spark MoE 加载静默挂死；`NCCL_NVLS_ENABLE=0` |
| #2334 | RoCEv2 高频小传输 stray completion → 集体死锁 |
| #2213 | 双 Spark ~1.1M 次 collective 后 SIGSEGV |
| PR#2202 | GB10 UMA 误判致 AR 数据损坏（已修） |
| #1946 | tree-graph 搜索失败 → nChannels=1 |
| #2345 | Ring/Tree 都会建跨 rail 链路，CROSS_NIC=0 拦不住 |
| — | HCA 名大小写（`roceP2p1s0f0` 的 P 必须大写，小写静默单 rail）；GID index 3 依赖 IPv6 link-local；6 节点 routed RDMA 不可行（QP RTR 超时，RoCE 需 L2 直连） |

### 6.7 其他传输
autoscriptlabs/nccl-mesh-plugin（~1500 行 C 自定义 NCCL net 插件，3 机 mesh 8+ GB/s @100G）、eugr/sparkrun（ring 注入 NET_PLUGIN=none/SUBNET_AWARE_ROUTING/MERGE_NICS=0；3 节点 ring 需专门 NCCL 分支否则静默 SIGTERM）、macncrash/sparkwire（RDMA over USB4 原型）。

---

## 7. 拓扑横向对比与实测数据点

| 拓扑 | allreduce 带宽 | 小消息延迟 | 实测 serving | 故障域 | 成本 |
|---|---|---|---|---|---|
| 4 线环双 rail（主流） | busbw ~26.6 GB/s 模型；84 MiB 6.56 ms | NCCL ring 90.6 µs / oneshot 16.9 µs | 见 §3.1 | 任一线断环变链 TP 全挂 | 4 根 DAC |
| 3 机三角 | 全直达无对角 | oneshot 直达 | 4 环比 3 三角 decode +14–27%（含第 4 GPU 归因） | 断一线退化为 2+孤儿 | 3 根 |
| hairpin 逻辑全互联 | 对角不新增带宽 | 小 ops 双路径 | RoCEnante serving | 依赖 marker 进程；fail-stop | 4 根 |
| 交换机（CRS804/812） | 线缆层同 196 Gb/s | — | MiaAI switched 口径 | 星型单点=交换机 | $1.1–1.5k+缺货 |
| 2 机直连 | 196 Gb/s（bond 后） | 最低 | — | 单线 | 1 根 |

数据点汇总：物理 200GbE→ib_write_bw 197 Gb/s→NCCL all_reduce 10.2 GB/s→sendrecv 9 GB/s→vLLM 的四层衰减框架（multimodalflow）；8 机 CRS804 NCCL 峰值 ~17.6 GB/s ≈140 GbE〔摘录〕；本机锁频 2550MHz +2-3%；阵列里唯一"同 repo 内 A/B"：alexellis NCCL v0.0.1 vs hardened（±2% 平手）、jnardiello E37 butterfly NO-GO（**环上 NCCL ring 实测最优**）。

---

## 8. 成本面

- 整机：DGX Spark 128GB $3,999→$4,699-6,950（涨价）；**ASUS GX10 $2,999**；国行 ¥37,999；64GB 版 $4,999。
- 交换机：CRS804-4DDQ（4×400G）$1,295/街价 ~$1,100，**长期缺货**（backorder 到 11 月）；CRS812 同病；SN2700 二手 ~$2k（仅 100G 接不了 200G 口）；SN3700 32×200G 翻新 $21,812。
- 线缆：第三方 200G DAC 0.5m $45–54；Amphenol 认证 $159/根；4 机环 = 4 根 ≈ $200–640。
- 整包方案商：OpenZeka Quad（CRS812 compute + CRS312 管理 + NAS）、Exxact、Petronella（线缆+CC BY 4.0 组网指南）。

---

## 9. 与本地方案对照（含本次核实）

**相同**：我们 = 4 线环双 rail + ring-only patched NCCL + RoCEnante-off + 前门 LB/key/Grafana——与 alexellis/jnardiello/knapcio-switchless 同构；我们的冷 prefill 3,064-3,470 处在生态第一梯队（SparkRing 3,500 同档）；qeval 73-75 与各家质量门同档。

**本次核实到的三个事实（笔记要修）**：
1. 笔记"双平面 8 根线"→ 实为 **4 根线×每线 2 rail**（本机 `ibdev2netdev` 证实每台 4 function 各 1 port）。
2. 笔记未记录：`.env.switchless` 已是 **knapcio `TRANSPORT=switchless` 机制**（SHA 9ddd77c2 = 我们的 SparkRing dual-domain 构建；`SWITCHLESS_ADDR_RANGE=10.200.0.0/16` 参数化寻址，无 othexmr 硬编码字面量）。我们的部署 = knapcio switchless 机制的**硬件认证实例**（上游作者自己没跑过环）。
3. knapcio 上游 2026-09-29 rebase（3ab5ca0）作者标注"未硬件认证"——我们 10-06 的部署与验收（八条+qeval 73-75）恰好补上这一环。

**我们独有/领先**：1M 上下文 + 40GiB KV（6.4M tokens，大于 tonyd2wild 1M-KV 仓的 3.45M 与 SparkRing GLM 的 6.13M）；锁频 2550MHz；Grafana/Prometheus 实时面板；前门 key+Anthropic 网关+Tailscale。
**我们缺的**：SparkCache 磁盘 KV 重放（64K 40K tok/s 级）、RoCEnante 环上化（knapcio 说明 DeepSeek recipe 有 ring-capable RoCEnante 但未移植到 GLM——"a port of roce/glm_roce is welcomed as PR"）、Rigmark 同口径基线。

---

## 10. 行动建议（优先级）

1. **立即（零风险）**：修正两处笔记（§9 的 1、2）；把 spark-bf94 的 `ibdev2netdev`/rail 带宽事实写进方案总结。
2. **立即（高价值）**：向 knapcio 上游开 issue/PR 回报 "switchless ring hardware-qualified on 4×Spark, qeval 73-75, transport notes"——全生态只有我们能做这个认证；同时可询问 othexmr 四 PF build 与我们 dual-domain build 的差异。
3. **短期**：跑 Rigmark（alexellis/rigmark，E36 对齐 flags）存档基线；考虑提交 Spark Arena 榜单（mpfaffenberger 的 GLM NVFP4 已在榜）。
4. **短期（运维）**：把 `ib_write_bw -b -x 3` 双向 ≥190 Gb/s 纳入巡检/Grafana 告警（线缆静默劣化是已证实的生产事故形态）；确认 `NCCL_NVLS_ENABLE=0` 已设。
5. **中期（选做）**：借鉴 christopherowen 的均衡 channel 策略（2 MiB AR -27%）做一次我们环上的 A/B；评估 SparkCache/磁盘 KV 重放（对我们 1M 长上下文重复前缀场景收益最大，但绑 R10/kraken 镜像生态）。
6. **中期（扩模型，注意与现集群互斥 ~110GiB/台）**：DeepSeek-V4.1-Flash 环上现成（LuZ/knapcio-DSV41）；Qwen3.8/MiMo 用 SparkRing 一键；gpt-oss 走 PP。商用部署先解决 DFlash2 NC-ND（canada-quant/G 或 MTP）。
7. **观望**：TensorFold TP4 环成熟度、b12x→FlashInfer 上游化、GPU-initiated networking（若解锁 dmabuf/peermem，全行业 host-relay 架构重写）、8 机 TP8。

---

## 11. 附录：未确认项与存疑项

- ZackO2o 跨项目对照表：直读两次超时，数字来自 agent 摘录（harness 差 20-25% 的警告为原文）。
- SparkRing "R10 镜像" 名称：全库检索无 R10，当前代次为 r8/kraken dev——我们笔记里的"R10"疑为口误或 jnardiello 私有称呼。
- MiaAI TP4 两套数字（45.4-134.2 vs 87.7-342.7）并存，疑为新旧 profile 版本，未逐一溯源。
- hawkBearPig/dgpp（105★）README 多次抓取失败；colonel-otto 3 机系列正文未核；localaiguyy 仓可能为 Apache Spark 噪音。
- "8×Spark CRS804 跑满 1.6 Tbps" 的 X 原帖未核验；CRS812 价格未拿到；800G 交换机入门价未查到。
- 超擎数智中文 NCCL 基准正文数值未完整抓取；HF DGX Spark Handbook ring 章节全文未复核。
- alexellis "How and Why We Bought 4x DGX Sparks" 无明确日期戳（推断 2026-10-06/07）。
- alexellis/switchless-nccl：README 称 v0.0.1 release，GitHub API 无 release 对象（downloads disabled），以 README 为准。
