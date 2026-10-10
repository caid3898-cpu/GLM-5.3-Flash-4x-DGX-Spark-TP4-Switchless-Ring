# GLM-5.3-Flash vLLM TP4 环网部署方案总结（供他人复现参考）

> 整理：2026-10-06 · 环境：4× DGX Spark (GB10, sm_121, 128GB 统一内存) · 仓库：`knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` @ 770d115
> 本文档面向想**在自己的四台 Spark 环网上复现这套部署**的人。写满了我们踩过的坑——它们比成功路径更值钱。

---

## 0. 一句话方案

把 knapcio 的 GLM-5.3-Flash vLLM TP4 配方（NVFP4 experts + lossless8 dense + DFlash2 投机解码）从"交换网+RoCEnante"移植到**四台无交换机的 200G 环网**上：SparkRing 的 switchless NCCL 补丁（强制纯 ring 算法，环上无树可建）+ dual-pci-domain 四端口构建 + 前门 LB。实测到达交换网方案同口径的 ~95%（code c1 打平）。

## 1. 硬件与拓扑前提

- 四台 DGX Spark，每台 2 个 QSFP 口（Socket Direct：每个物理口 = 2 个 PCIe root 的各一个 function，共 4 个 RDMA 设备/rocep1s0f0,f1 + roceP2p1s0f0,f1）
- **每根线的两条 Socket-Direct rail 都用**（4 根线、共 8 条逻辑链路，每台 4 个 RDMA function 全配 IP）；只用主域单 rail（2 口旧库）也能跑但 prefill 慢 ~11%。〔2026-10-07 校正：每台只有 2 个物理 QSFP 口，原稿"每对邻居 2 根线、共 8 根"是误记——本机 `ibdev2netdev` 实证每台 4 function 各 1 port〕
- 环序按物理接线排 rank0→3，每台 f0 口朝下一跳、f1 口朝上一跳
- 每条 rail 一个 /30（如 10.200.{1,3,5,7}.0/30 为主域 rail、{2,4,6,8} 为次域 rail），**每根线的 rail 落在各自不同的 /24**（`SWITCHLESS_SUBNET_PREFIX_LEN=24` 据此区分每根线），MTU 9000
- 管理网独立（我们 192.168.31.0/24，NCCL bootstrap 走它，数据面绝不经过）

## 2. 关键组件与版本（全部实测）

| 组件 | 版本/来源 | 说明 |
|---|---|---|
| 底座镜像 | `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef...` | 48 层，bf94/6e8e 拉取会被 ghcr denied → 用 `docker save\|load` 分发 |
| RoCE 镜像 | 仓库 `Dockerfile.roce` 构建 `glm53-roce:v11-b58f34ea` | 注意：**docker load 会丢 RepoDigests**，在这类节点上构建要 `--build-arg BASE_IMAGE=<本地tag>`，否则 FROM @sha256 回源 registry 报 denied |
| NCCL | 2.30.7 + `nccl-2.30.7-dual-pci-domain.patch`（SparkRing f16b5f4，累计补丁含 switchless-cycle） | `make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"` 约 3 分钟；2 口旧库 `0b9e4888...` / 4 口新库 `9ddd77c2...` |
| 权重 | `nvidia/GLM-5.3-Flash-NVFP4` @ 09b04e5e，44 文件 204.5GB | 必须**逐文件和 HF API 校验尺寸**（见 §5 坑 2） |
| 转换 | 仓库 `scripts/build_lossless8.sh` + `drafter_fp8.py`，四台各跑一次 | 产出 `~/models/glm-quant-mix/lossless8`（硬链接布局，不可跨机 rsync）+ `GLM-5.3-Flash-DFlash2-fp8blk` |
| GPU 频率 | `nvidia-smi -lgc 2550`（已写入 systemd 永久化） | +2~3%，温度 47-52°C 很安全 |

> **2026-10-07 补记（启动机制溯源）**：部署用的是 knapcio 上游 2026-09-29 起内置的 `TRANSPORT=switchless`（社区 PR #1，作者 othexmr；本仓库 commit 770d115 即该发布）。我们的 NCCL 属 SparkRing dual-pci-domain 系：`strings` 实证**不含** othexmr 四-PF build 硬编码的 `10.100.224-227.0/24` 字面量，靠参数化 `SWITCHLESS_ADDR_RANGE=10.200.0.0/16` + `NCCL_IB_SUBNET_AWARE_ROUTING=1` + `EXTENDED_IPV4_GIDS/PRESERVE_PCI_DOMAIN` 正常工作，四 PF 全列 `IB_HCA` 可用。上游作者自己的机群在交换机后、未在环上启动过该模式——**本部署即该机制的硬件资格认证**（qeval 73-75/75；prefill 冷 3064-3470 tok/s；2026-10-07 起双向 ib_write_bw 巡检 8×rail ≈203 Gb/s）。

## 3. 部署流程（顺序很重要）

```bash
# 1. 每台：仓库 + .env.switchless（IB_HCA 写 4 个口）+ 补丁 NCCL 目录 + 权重(~/models/nvidia/...)
# 2. 每台：构建 RoCE 镜像（见 §2 的 build-arg 坑）
# 3. 每台：转换权重（容器挂载 ~/models，NVFP4 与输出必须同一文件系统供硬链接）
# 4. bf94（head）：修好对四台的免密 ssh（~/.ssh/config 映射 spark-* → IP + keyscan），
#    注意 ./start.sh 的 DRY 模式不做真实 ssh——dry-run 全 PASS 不代表真实模式能跑
# 5. bf94：ENV_FILE=.env.switchless ./start.sh serve
# 6. 等 /health 200（冷启动 ~9.5 分钟，暖缓存 ~2.5 分钟）
# 7. 前门 LB（systemd --user）：GLM_BACKENDS={"A":"http://127.0.0.1:8093"}，GLM_MODEL=GLM-5.3-Flash-FP8
```

## 4. 最终性能（2400→2550MHz，仓库口径）

- sparkDash：prose c1 71.3 / code c1 115.1（与 README 交换网 119.8 打平）/ structured 142.1 / json 106.9；prose 聚合 c4 148 / c8 219 / c16 279
- prefill 冷：8k-128k 全在 3064-3470 tok/s（高于 README@2200 的 3222）
- accept_probe：prose 39.2ms·58.1 / code 96.0 / json 117.9 tok/s
- qeval 质量门：73-75/75（与仓库声明的 74-75 窗口一致）

## 5. 坑（按踩坑成本排序）

1. **下载脚本的假完成**：`get_weights.sh` 被中断后重跑，`[ -s file ]` 会跳过**截断文件**——DONE+fails=0 但总量少了 33GB。务必逐文件比 HF API 的 size（我们写了 `verify_weights.sh`）。
2. **start.sh DRY 模式不做真实 ssh**：preflight 全 PASS 但真实 serve 在 ssh 第一步就挂（bf94 解析不了其他节点名）。所有"dry 通过"结论先打问号。
3. **docker load 丢 RepoDigests** → build 回源 denied（§2）。
4. **转换的硬链接/软链接陷阱**：HF 缓存 snapshot 全是软链，容器内悬空；symlink 方案跨挂载硬链接 EXDEV。正解：权重 `mv` 进 `~/models/nvidia/`（同 fs 秒级）、drafter `cp -rL` 实体化、整树单挂载点。
5. **rsync 目标父目录不存在会整跳失败**（只报一个 code 11），分发前先 `mkdir -p`。
6. **节点间 ssh 要按环向建**（0167→6e8e→bfe5→bf94），bfe5 私钥在但 .pub 丢了，用 `ssh-keygen -y` 重建。
7. **无 sudo 环境的全套替代**：特权容器挂宿主 /usr/sbin（注意宿主 `ip` 是指向 /bin/ip 的软链，要单独挂）、`/proc/sys` 写 sysctl、SYSTEMD_IGNORE_CHROOT=1 + --pid host 管 systemd --user。
8. **权重转换容器拒绝已存在的输出目录**：`os.mkdir` 语义，重跑前把残留目录改名（mv 到 *.vNfail，别 rm -rf，网关会拦）。

## 6. 调优结论（都做了实测，别重复踩）

| 尝试 | 结果 |
|---|---|
| dual-pci-domain 4口 NCCL | **prefill +11%，decode 持平——保留** |
| SparkRing mesh + RoCEnante（硬件 hairpin 全互联） | 完整移植成功（4-rank 测试全绿，热循环 25µs vs NCCL 51µs），但**在 serving 上下文 one-shot 84µs > NCCL 59µs**（对到达不齐敏感）——生产关闭，MESH_ROCE=0 一键可切 |
| NCCL 2.31.2 自构建 | 与 2.30.7 完全持平，回退 |
| MIN/MAX_NCHANNELS=4 | 持平（与 knapcio DeepSeek 文档的 A/B 一致） |
| T1 实测重标定截断代价表 | 本机行长成本仅 0.5ms（fleet 表假设 2-3ms），但 policy 的 host 决策税使长验证步长爆炸；fleet 默认反而最优 |
| CUDA_MODULE_LOADING=EAGER 修 devselect | devselect 需要 GPU 条件图捕获，本平台 CUDA 13 不支持（私有池/EAGER 都试过）；EAGER 还让启动 +20 分钟。放弃（损失 README 声称的 +1.9% prose） |
| GPU 2400→2550MHz | **+2~3%，保留** |

**方法论一句话**：没修量具就别下结论——profile 出"96 次 allreduce×59µs=5.7ms/步"之前，我们一度以为差距在带宽/频率/acceptance，全错。

## 7. 回退

- 旧 2×TP2 TensorFold 完整保留在 `~/glm53-tf`（四台），启动方式见旧笔记
- 前门 LB 改回旧后端：drop-in `~/.config/systemd/user/glm53-lb.service.d/tp4.conf` 删掉即可
- 2口 NCCL（`~/nccl-2.30.7-switchless`）与 env 备份 `.env.switchless.2port.bak` 都在

## 8. 监控与运维速查

- `cd ~/glm53-flash-4x-spark && ENV_FILE=.env.switchless ./start.sh status|logs 0 400|stop|serve`
- 前门健康：`curl 192.168.31.21:8002/health`；负载：`/lb/stats`
- **环网带宽巡检（2026-10-07 起）**：`systemctl --user start glm-fabric-check.service`（每 6h 自动，`glm-fabric-check.timer`）；结果 `~/glm53-ops/fabric-check/{history.log,latest.json}`；每 rail 双向 ib_write_bw ≥190Gb/s 为健康（实测 ~203）；vLLM 有在跑请求时自动跳过。脚本与 unit 本地留档：`/Volumes/cd/zcode/glm53-ops/`
- 看板：bf94 `:5555`（sparkDash，已改指 8093）
- 验收八条（doctor/preflight、四容器、环网日志三要素、对话、qeval×3、accept_probe、prefill、频率标注）全部通过的做法见部署笔记
