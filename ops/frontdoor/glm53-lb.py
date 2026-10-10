#!/usr/bin/env python3
# glm53-lb.py —— 双集群前门负载均衡器（2026-10-01）
#
# 目的：把两个 2 节点 GLM 集群藏在**一个不变地址**后面：192.168.31.21:8002
#
#   客户端 ──▶ :8002 (本服务) ──┬──▶ 集群 A  bf94 :8888   (GLM-5.3-Flash-EXL3)
#                               └──▶ 集群 B  6e8e :8888   (GLM-5.3-Flash-EXL3)
#                    ▲
#                    └── :8003 responses-gateway (Anthropic→OpenAI，本服务不动)
#
# ── 为什么是「哈希粘滞」而不是轮询 ──────────────────────────────────────
# 轮询会把同一段对话的第 1 轮和第 2 轮甩到不同集群，两边的前缀缓存都命中不了，
# 每轮都要重算全量 prompt。仓库里 SHARED_PREFIX=1 依赖「同一个 system prompt
# 留在同一台机器上」，粘滞路由正好与它同向。
#
# 粘滞键 = hash( system prompt + 首条用户消息 )：
#   这两项在一段对话的多轮之间**逐字节不变**，而第 2 轮之后的消息一直在变，
#   所以它是唯一既能跨轮稳定、又能把不同对话打散的选择。
#
# ── 「均摊」的粒度是「对话」不是「请求」 ────────────────────────────────
# 不同对话的哈希散列到两个集群，长期看 50/50。同一段对话粘在一台，
# 是为缓存命中做的显式取舍，不是遗漏。
#
# ── 降级 ───────────────────────────────────────────────────────────────
# 对端不健康是**最高优先级**，覆盖下面的一切；它恢复后自动回到原集群（缓存重新命中）。
#
# ── 负载感知（2026-10-02 新增）─────────────────────────────────────────
# 纯一致性哈希在**小批量相似任务**上会成片倒向一边：实测 6 个不同任务 5/6 落在同一台，
# AutoReel 暂停后重启的那批任务就是这样把一台塞满、另一台空转的。
# 但**不能**直接把哈希换成「按在途最少选」——那会把一个会话在两台之间来回甩，
# 前缀缓存全丢（同会话多轮不再命中）。
# 正确做法是**只在键首次出现时**用负载选，之后一律走记住的那台：
#   · 首次出现 ⇒ 本机上还没有它的缓存，丢不丢都无所谓 ⇒ 允许为均衡而选
#   · 已经绑定过 ⇒ 缓存在那台上 ⇒ 必须回去，哪怕另一台更闲
# 绑定**落盘**（blake2b 键，不落对话原文），LB 重启后仍然回到同一台。
#
# 环境变量：
#   GLM_BACKENDS='{"A":"http://127.0.0.1:8888","B":"http://192.168.31.28:8888"}'
#   GLM_MODEL='GLM-5.3-Flash-EXL3'
#   GLM_HEALTH_INTERVAL=10
#   GLM_STICKY_PATH='/home/dada/glm53-lb/sticky.json'   绑定落盘位置
#   GLM_STICKY_MAX=20000                                绑定条数上限（超出丢最久未用的）
#
# 启动：uvicorn glm53-lb:app --host 0.0.0.0 --port 8002

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# build 标记：改动后**必须**同步 bump 一次。
# 存在的唯一理由是让自检能发现「我测的还是旧文件」—— 那种情况下测试会全绿，
# 而结论完全无效（踩过：scp 的是 glm53-lb.py.new，被测的却是 glm53_lb.py）。
BUILD = "2026-10-07-closelegacy"
BACKENDS: dict[str, str] = json.loads(os.environ.get(
    "GLM_BACKENDS", '{"A": "http://127.0.0.1:8888", "B": "http://192.168.31.28:8888"}'))
# Anthropic 协议没有后端支持（TensorFold 不提供 /v1/messages），
# 但 omp / claude-code 把 ANTHROPIC_BASE_URL 指向了 :8002（过去靠 SGLang 原生支持）。
# 所以 :8002 收到 Anthropic 请求时转交给本机 8003 的 responses-gateway 做协议转换，
# 它再用 OpenAI 协议回调 :8002 —— 那一次会被下面的正常路由逻辑送到某个集群，环路到此终止。
ANTHROPIC_GATEWAY = os.environ.get("GLM_ANTHROPIC_GATEWAY", "http://127.0.0.1:8003")
REAL_MODEL = os.environ.get("GLM_MODEL", "GLM-5.3-Flash-EXL3")
# 兼容旧客户端名：响应式网关与所有工具配置里写死的是 deepseek-v4.1-flash
LEGACY_MODELS = {
    "GLM-5.3-Flash-EXL3",  # 2026-10-06 旧 TensorFold 集群名 → 新 TP4
    "GLM-5.3-Flash",
    "deepseek-v4.1-flash", "deepseek-v4-flash", "local-v4-flash",
    "deepseek-v4-flash-vision-exp",
}
# 3 秒而不是 10：饱和可能只持续十几秒，10 秒采样会整段错过（实测并发 12 的
# 高峰只持续 ~30 秒，而判据要靠 running 抓到它）。代价只是多几次本地 GET。
HEALTH_INTERVAL = float(os.environ.get("GLM_HEALTH_INTERVAL", "3"))
STICKY_PATH = os.environ.get("GLM_STICKY_PATH", "/home/dada/glm53-lb/sticky.json")
STICKY_MAX = int(os.environ.get("GLM_STICKY_MAX", "20000"))

# ── 深度探针 + 卡死掐流（2026-10-05）────────────────────────────────────
# 为什么 /health 不能单独当判据：那天实测**卡死的引擎 /health 照样 200**，
# requests_running 甚至报 0，而同一时刻真实 completion 90 秒零字节、
# rank0 CPU 自旋到 115%。判据撒谎的代价是整条链跟着撒谎 —— LB 往死集群发流量，
# memguard 也把它判成健康，于是既不摘流量也不重启，ZCode 只能一直「重连中」。
# 所以这里补两个东西：① 定期发**一发真实 completion** 验「能不能解码」；
# ② 流式回源加**无进展超时**，掐掉那种「收下请求却永不吐字」的连接。
#
# 探针要 max_tokens=1：要的不是内容，是「请求能不能走完整个解码循环」。
# 卡死的引擎会收下请求（requests_total +1）却不吐字，所以这一步才有鉴别力。
DEEP_INTERVAL = float(os.environ.get("GLM_DEEP_INTERVAL", "8"))
DEEP_INTERVAL_DEAD = float(os.environ.get("GLM_DEEP_INTERVAL_DEAD", "20"))
# 全都健康**且**最近 IDLE_BACKOFF_AFTER 秒内没有任何转发流量时，把探针降频。
# 理由：空闲时没有任何东西会突然卡死（卡死几乎都由真实请求触发），没必要烧电；
# 而一旦有流量、或有一台不健康，立刻回到高频 —— 检测延迟只在真正需要时才付。
DEEP_INTERVAL_IDLE = float(os.environ.get("GLM_DEEP_INTERVAL_IDLE", "45"))
IDLE_BACKOFF_AFTER = float(os.environ.get("GLM_IDLE_BACKOFF_AFTER", "60"))
# 两台探针之间的错峰秒数
DEEP_STAGGER = float(os.environ.get("GLM_DEEP_STAGGER", "3"))
DEEP_TIMEOUT = float(os.environ.get("GLM_DEEP_TIMEOUT", "10"))
# 超过这个秒数没验过就不敢发流量 —— 宁可少发，也不要发进一个没验过的黑洞。
# 必须**明显大于**最长的探针间隔（DEEP_INTERVAL_IDLE=45）：我第一版把它也设成 45，
# 于是空闲态下 age 恒在 45.1s 上下抖动，集群被判成「探针过期」而无故掉线
# （日志里实测到 health B: True -> False ... age=45.2s）。留 2.7 倍余量。
DEEP_STALE = float(os.environ.get("GLM_DEEP_STALE", "120"))
DEEP_FAIL_TOL = int(os.environ.get("GLM_DEEP_FAIL_TOL", "2"))
# 流式回源多久没有任何字节就判定为死锁并掐掉。掐流的**首要目的**是释放 lane
# （那是泄漏的根源），30s 静默已经远超任何合法 prefill（实测最长约 3.8s）。
# 万一误判，深度探针会在一个周期内把它重新摘回来，代价有界。
STREAM_IDLE = float(os.environ.get("GLM_STREAM_IDLE", "30"))
# 静默多久才彻底放弃这条流。给得宽是因为「多数卡死会自愈」，而真正死锁的那次
# 通常在 memguard 判定后重启引擎（约 2~3 分钟）时就会被连带清掉，这里只是兜底。
STREAM_GIVEUP = float(os.environ.get("GLM_STREAM_GIVEUP", "600"))

# ── 饱和判据（2026-10-04 23:30 换掉旧的不均衡判据）────────────────────
# 旧判据：IMBALANCE_ABS/REL 双阈值，比的是 **LB 侧在途数**的绝对差与相对比。
# 为什么废掉：每台 --parallel 4，只有 4 条 lane。实测（lb_stress.py，2026-10-04）
# 真实搬台发生在 inflight = (0,3) / (3,1) / (3,0) 这些组合上 —— 也就是
# 「3 条 lane 在跑、另一台 0~1 条」的**完全健康状态**。
# 换算成代价：一次对话平均换台 2~4 次，每次把 85k token 的历史重算一遍，
# 留原台 1.31s vs 搬台 104.44s —— 每轮白付 24~48 秒。
# 即「负载均衡在把已经很贵的缓存反复打碎，而系统从来没饱和过」。
#
# ⚠️ 换判据时踩的第二个坑：**requests_waiting 恒为 0，不能用它判饱和。**
# 高频采样（0.7s 一次，压到并发 12）实测：cluster A 的 requests_running 冲到
# 10/11/12，而 requests_waiting 19 次采样**全是 0** —— TensorFold 即便超额
# 并发也不把请求放进 waiting 队列。所以第一版改法（判 waiting>=1）等于
# 「永远不搬」，把负载均衡整个关掉了。这个坑只有专门去测「该触发时会不会触发」
# 才暴露得出来 —— 只测「不该搬的时候没搬」的判据全是绿的。
#
# 现在的判据：**running 相对本机 lane 数**。running >= lanes 即视为跑满。
LANES_DEFAULT = int(os.environ.get("GLM_LANES", "4"))
# 2.0 而不是 1.0：实测并发 4 时 A 的 running 就能顶到 4（正好等于 lanes），
# 用 1.0 会在轻负载下就开始搬（实测 120 个请求搬了 12 次）。
# 2.0 = running >= 8 才算真过载：并发 4 永远到不了，并发 12 时 A 实测能到 10~12。
SAT_OVERSUB = float(os.environ.get("GLM_SAT_OVERSUB", "2.0"))  # running >= lanes*该值 即饱和
MOVE_COOLDOWN = float(os.environ.get("GLM_MOVE_COOLDOWN", "300"))  # 搬过之后 5 分钟内不再搬

# 旧判据保留：/health 仍展示它，方便对照历史数据；choose() 已不再使用。
IMBALANCE_ABS = int(os.environ.get("GLM_IMBALANCE_ABS", "2"))
IMBALANCE_REL = float(os.environ.get("GLM_IMBALANCE_REL", "2.0"))

app = FastAPI(title="glm53 dual-cluster front door", docs_url=None, redoc_url=None)

_client: Optional[httpx.AsyncClient] = None
_health: dict[str, Optional[bool]] = {n: None for n in BACKENDS}
# 浅探活（GET /health）的**原始**结论，只作观测对照；路由读的是 _health（合成后）
_shallow: dict[str, Optional[bool]] = {n: None for n in BACKENDS}
# 深度探针状态：ok=最近一次真实 completion 是否成功，at=探针时刻，
# fails=连续失败次数，ok_at=最近一次成功时刻，ms=最近一次耗时
_deep: dict[str, dict] = {n: {"ok": None, "at": 0.0, "ok_at": 0.0,
                              "fails": 0, "ms": None, "err": ""} for n in BACKENDS}
_reaped = 0         # 被「无进展超时」掐掉的回源流数
# 后台任务句柄：留着才能回答「探针循环是不是还活着」。它静默死掉是最坏的一种故障
# —— 进程看起来完全正常，但从此再也不检测卡死。
_deep_task: "Optional[asyncio.Task]" = None
_health_task: "Optional[asyncio.Task]" = None
# 每台最近一次真正转发请求的时刻，用来判断「现在是不是完全空闲」→ 探针要不要降频
_last_routed_at: dict[str, float] = {n: 0.0 for n in BACKENDS}
# 每台最近一次因对端不健康而被绕过的次数（验收用）
_bypassed: dict[str, int] = {n: 0 for n in BACKENDS}
_stats = {"routed": 0, "bypassed": 0, "errors": 0, "by_cluster": {n: 0 for n in BACKENDS}}

# ── 负载感知 + 持久粘滞（见文件头注释）──────────────────────────────
# _sticky: blake2b(会话键) -> 集群。LRU，_sticky_max 之外丢最久没用的。
_sticky: "OrderedDict[str, str]" = OrderedDict()
# 每台当前在途请求数：转发时 +1，流结束/响应返回时 -1。比轮询准，没有滞后。
_inflight: dict[str, int] = {n: 0 for n in BACKENDS}
_sticky_dirty = False
_last_save = 0.0
_rebound = 0          # 因饱和而**真正**换了台的次数（target != bound 才计）
_gate_fired = 0       # 饱和判据触发的总次数（含被冷却期挡下的）
_cooldown_blocked = 0 # 因冷却期没搬成的次数
# 集群自己报的排队情况：{"A": {"running": n, "waiting": m}, ...}
# 这是判断「真饱和」的唯一依据 —— LB 侧在途数是瞬时噪声。
_sload: dict[str, dict] = {n: {"running": 0, "waiting": 0, "lanes": None} for n in BACKENDS}
# 每个会话上次被搬走的时间，用于迟滞
_last_move: dict[str, float] = {}


def _sticky_id(key: str) -> str:
    """绑定表的键：会话键的 blake2b。**不落对话原文**。"""
    return hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=12).hexdigest()


def load_sticky() -> None:
    try:
        raw = json.loads(Path(STICKY_PATH).read_text())
    except Exception as e:
        print(f"[lb] sticky: 无可载入的绑定（{type(e).__name__}）", flush=True)
        return
    for k, v in (raw.get("bindings") or {}).items():
        if v in BACKENDS:
            _sticky[k] = v
    print(f"[lb] sticky: 载入 {len(_sticky)} 条绑定 ← {STICKY_PATH}", flush=True)


def save_sticky(force: bool = False) -> None:
    """落盘绑定。默认 2 秒内不重复写（去抖），原子替换。"""
    global _sticky_dirty, _last_save
    if not _sticky_dirty:
        return
    now = time.time()
    if not force and now - _last_save < 2.0:
        return
    try:
        p = Path(STICKY_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"bindings": dict(_sticky)}))
        tmp.replace(p)                      # 原子替换：读侧永远看到完整文件
        _sticky_dirty = False
        _last_save = now
    except Exception as e:
        print(f"[lb] sticky 落盘失败: {e}", flush=True)


# ── 粘滞键 ──────────────────────────────────────────────────────────────
def _text_of(content: Any) -> str:
    """把 OpenAI 的各种 content 形态压成一段稳定文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif part.get("type"):
                    # 图片/视频 part 只取类型标记，避免 base64 全文进哈希
                    out.append(f"<{part['type']}>")
        return "".join(out)
    return "" if content is None else str(content)


def session_key(body: dict) -> str:
    """一段对话的稳定标识。多轮之间逐字节不变。"""
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        head = []
        for m in msgs[:1]:                      # 首条（通常是 system）
            if isinstance(m, dict):
                head.append(_text_of(m.get("content")))
        for m in msgs:                          # 显式的 system 角色
            if isinstance(m, dict) and m.get("role") == "system":
                head.append(_text_of(m.get("content")))
                break
        for m in msgs:                          # 首条 user
            if isinstance(m, dict) and m.get("role") == "user":
                head.append(_text_of(m.get("content")))
                break
        key = "\u0001".join(head)
        if key.strip():
            return key
    p = body.get("prompt")                      # /v1/completions 的裸 prompt
    if isinstance(p, list) and p:
        p = p[0]
    if isinstance(p, str) and p:
        return p[:4096]
    return ""


def pick_cluster(key: str) -> str:
    """一致性哈希 → 集群。空 key 也稳定落到同一台。"""
    if not BACKENDS:
        raise RuntimeError("GLM_BACKENDS 为空")
    h = hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=8).digest()
    return sorted(BACKENDS)[int.from_bytes(h, "big") % len(BACKENDS)]


def _lanes(name: str) -> int:
    return (_sload.get(name) or {}).get("lanes") or LANES_DEFAULT


def _saturated(name: str) -> bool:
    """这台是不是**真的跑满了**。

    判据用集群自己 `/health` 报的 `requests_running` 相对本机 lane 数，
    **不用 requests_waiting** —— 实测那个字段恒为 0（见文件头的坑）。
    也不用 LB 侧在途数 —— 那是瞬时噪声，「3 条 lane 在跑」会被误判成失衡。
    """
    s = _sload.get(name) or {}
    return s.get("running", 0) >= _lanes(name) * SAT_OVERSUB


def _has_room(name: str) -> bool:
    """这台还有空位（running 不到 lane 数）。"""
    s = _sload.get(name) or {}
    return s.get("running", 0) < _lanes(name) * SAT_OVERSUB


def _in_cooldown(sid: str) -> bool:
    """这个会话是不是刚被搬过。迟滞是为了挡住「A→B→A→B」那种秒级来回。"""
    last = _last_move.get(sid)
    return last is not None and (time.time() - last) < MOVE_COOLDOWN


def _imbalanced(healthy: list[str]) -> bool:
    """⚠️ 已弃用：旧的「在途数双阈值」判据。choose() 不再用它，只在 /health 里展示作对照。

    历史行为：每台 --parallel 4，inflight 差 ≥2 且高者 ≥2× 低者就算倾斜。
    实测这个判据在并发 ≥3 时几乎恒真（(3,1)/(3,0)/(0,3) 全都算），
    把「正常忙」当成「失衡」，于是每个请求都把已绑定的会话搬走。
    真实代价见笔记 §10.5：留原台 1.31s，搬台 104.44s。
    """
    if len(healthy) < 2:
        return False
    loads = [_inflight.get(n, 0) for n in healthy]
    lo, hi = min(loads), max(loads)
    return (hi - lo) >= IMBALANCE_ABS and hi >= IMBALANCE_REL * max(lo, 1)


def choose(key: str) -> tuple[str, bool]:
    """返回 (集群名, 是否因对端不健康而降级)。

    顺序（后者只在更靠前的情况不成立时才生效）：
      1. **两台均衡时，已绑定的会话 → 回绑定的那台**（缓存优先，零代价）。
      2. **两台明显不均衡时 → 跟随最闲的那台，并把绑定改写到新家**。
         改写之后这段对话的缓存跟着搬过去，只冷一次；此后它在新家的绑定又稳定了。
         依据：前缀缓存在**每台各自持有**（实测 A/B 对同一个 3200-token system prompt
         都能 cached=3200、prefill 0.158 vs 0.160s），共享前缀天然两边都有，
         真正只在单台的只有这段 job 自己的对话历史 —— 挪台的代价只有它。
      3. 绑定的那台**不健康** → 解除绑定，降级到健康的一台（`degraded=True`）。
      4. **首次出现** → 按在途请求数最少的健康集群选；同数时用哈希打破平局。
      5. 选定后**落盘记住**，LB 重启后仍回到同一台。
    """
    healthy = [n for n in sorted(BACKENDS) if _health.get(n) is not False]
    if not healthy:                                   # 两台都探测失败：按哈希硬选
        return pick_cluster(key), False

    sid = _sticky_id(key) if key else ""
    bound = _sticky.get(sid)
    if bound is not None and bound not in healthy:    # ③ 绑定的那台挂了
        del _sticky[sid]
        bound = None

    hash_want = pick_cluster(key)
    # 同数时偏向哈希原本想去的那台：保留「同负载下结果与旧版一致」这个性质
    def rank(n: str) -> tuple[int, int]:
        return (_inflight.get(n, 0), 0 if n == hash_want else 1)

    global _sticky_dirty, _rebound, _gate_fired, _cooldown_blocked

    if bound in healthy:
        # ── 已绑定会话：默认**不动**，只有「真饱和 + 对端空闲 + 不在冷却期」才搬 ──
        # 2026-10-04 改。旧逻辑在这里用「在途数比值」，把 (3,1) 这种健康状态当失衡，
        # 结果每个请求都搬一次，85k 的对话来回搬，每轮白付 24~48 秒（笔记 §10.5）。
        others = [n for n in healthy if n != bound]
        busy = _saturated(bound)
        idle = bool(others) and all(_has_room(n) for n in others)
        if not (busy and idle):
            _sticky.move_to_end(sid)          # 常规路径：粘滞优先，零代价
            return bound, False
        _gate_fired += 1
        if _in_cooldown(sid):
            _cooldown_blocked += 1            # 刚搬过，别来回抖
            _sticky.move_to_end(sid)
            return bound, False
        target = min(others, key=rank)
        if target == bound:                   # 理论到不了，留个保险
            return bound, False
        _rebound += 1
        _last_move[sid] = time.time()
        print(f"[lb] 饱和迁移（{bound}: running={(_sload.get(bound) or {}).get('running', 0)}"
              f"/{_lanes(bound)} → {target}: running={(_sload.get(target) or {}).get('running', 0)}"
              f"/{_lanes(target)}）：会话 {sid[:8]} {bound} → {target}", flush=True)
        _sticky[sid] = target
        _sticky.move_to_end(sid)
        _sticky_dirty = True
        return target, False

    if hash_want not in healthy:                      # ③ 对端不健康 → 降级
        _bypassed[hash_want] = _bypassed.get(hash_want, 0) + 1
        target, degraded = min(healthy, key=rank), True
    else:
        # 新会话（没有绑定，搬它不损失任何缓存）→ 在途数最少优先，哈希破平。
        # 实测这个落点策略是对的：并发 1→8 时 A 占比 87.5%→72.5%→62.5%，
        # 并发 8→12 时 p90 从 9.36s 降到 6.96s、吞吐 1.68→2.40 req/s（笔记 §10.2）。
        target, degraded = min(healthy, key=rank), False

    if sid:                                           # ④/⑤ 落盘记住
        _sticky[sid] = target
        _sticky.move_to_end(sid)
        while len(_sticky) > STICKY_MAX:
            _sticky.popitem(last=False)               # 丢最久没用过的
        _sticky_dirty = True
    return target, degraded


# ── 生命周期 ────────────────────────────────────────────────────────────
async def _probe(url: str) -> tuple[Optional[bool], dict]:
    """探活并顺手采下负载。返回值带 running/waiting —— 饱和判据要用。"""
    try:
        r = await _client.get(f"{url}/health", timeout=3.0)
        ok = r.status_code < 500
        load = {"running": 0, "waiting": 0, "lanes": None}
        if ok:
            try:
                d = r.json()
                st_ = d.get("streams") or {}
                load = {"running": int(d.get("requests_running") or 0),
                        "waiting": int(d.get("requests_waiting") or 0),
                        "lanes": int(st_.get("max") or 0) or None,
                        "paused": int(st_.get("paused") or 0),
                        "filling": int(st_.get("filling") or 0),
                        "decoding": int(st_.get("decoding") or 0),
                        "pool_free_tokens": d.get("pool_free_tokens"),
                        "iteration_s": d.get("iteration_s"),
                        "busy": d.get("busy")}
            except Exception:
                pass                            # 拿不到负载不影响健康判定
        return ok, load
    except Exception:
        return False, {"running": 0, "waiting": 0, "lanes": None}


async def _health_loop() -> None:
    while True:
        try:
            for name, url in list(BACKENDS.items()):
                ok, load = await _probe(url)
                if ok != _shallow.get(name):
                    print(f"[lb] shallow {name}: {_shallow.get(name)} -> {ok}", flush=True)
                _shallow[name] = ok
                _sload[name] = load
            _reconcile()
            await asyncio.sleep(HEALTH_INTERVAL)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[lb] 浅探活轮次异常（已忽略，继续下一轮）: "
                  f"{type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(HEALTH_INTERVAL)


# ── 深度探针：验「能不能解码」，而不只是「进程活着」──────────────────
def _mark_dead(name: str, why: str) -> None:
    """把一台立刻判成不可用（不等下一个探针周期）。

    只在**已经有确凿证据**时调用：探针超时，或回源流无进展被掐。
    """
    d = _deep.setdefault(name, {"ok": None, "at": 0.0, "ok_at": 0.0,
                                "fails": 0, "ms": None, "err": ""})
    d["ok"] = False
    d["fails"] = max(int(d.get("fails", 0)), DEEP_FAIL_TOL)
    d["at"] = time.time()
    d["err"] = why[:160]
    _reconcile()


async def _deep_one(name: str) -> None:
    ok, ms, err = await _deep_probe(name)
    d = _deep.setdefault(name, {"ok": None, "at": 0.0, "ok_at": 0.0,
                                "fails": 0, "ms": None, "err": ""})
    d["ok"] = ok
    d["ms"] = round(ms, 1)
    d["err"] = err
    d["at"] = time.time()
    if ok:
        d["fails"] = 0
        d["ok_at"] = d["at"]
    else:
        d["fails"] = int(d.get("fails", 0)) + 1


async def _deep_probe(name: str) -> tuple[bool, float, str]:
    """发一发真实 completion。返回 (是否成功, 耗时ms, 错误摘要)。"""
    url = BACKENDS[name] + "/v1/chat/completions"
    body = json.dumps({"model": REAL_MODEL,
                       "messages": [{"role": "user", "content": "ok"}],
                       "max_tokens": 1, "stream": False}).encode()
    t0 = time.time()
    try:
        r = await _client.post(url, content=body, timeout=DEEP_TIMEOUT,
                               headers={"content-type": "application/json"})
        return r.status_code == 200, (time.time() - t0) * 1000, \
            ("" if r.status_code == 200 else f"http {r.status_code}")
    except Exception as e:
        return False, (time.time() - t0) * 1000, f"{type(e).__name__}: {e}"[:150]


def _reconcile() -> None:
    """把浅探活和深探针**合成一个**可路由结论，写进 _health。

    刻意只让 _health 暴露合成后的结论：choose() / /health / /metrics 全都读它，
    于是「别往卡死的集群发流量」只需要改这一处，不会漏掉某条路径。
    """
    now = time.time()
    for n in BACKENDS:
        d = _deep.get(n) or {}
        age = now - (d.get("at") or 0.0)
        # 没验过（新启动/刚重启）也算不可用：宁可先不发流量
        verified = d.get("ok") is True and int(d.get("fails", 99)) < DEEP_FAIL_TOL
        fresh = bool(d.get("at")) and age <= DEEP_STALE
        verdict = bool(_shallow.get(n)) and verified and fresh
        if _health.get(n) != verdict:
            print(f"[lb] health {n}: {_health.get(n)} -> {verdict}"
                  f" (shallow={_shallow.get(n)} deep={d.get('ok')}"
                  f" fails={d.get('fails')} age={age:.1f}s"
                  f" {d.get('ms')}ms {d.get('err')})", flush=True)
        _health[n] = verdict


async def _deep_loop() -> None:
    # 这个循环一旦抛异常就会**静默死掉**，之后再也不检测卡死 —— 而 LB 进程照样
    # 正常服务，看起来完全正常。所以整轮包 try/except：任何单轮出错都只记日志、
    # 继续下一轮，绝不让它退出。任务句柄也留着，好让 /lb/stats 能报它是否还活着。
    while True:
        try:
            # 错峰：两台不要在同一瞬间各吃一发。TP2 的两个 rank 共用同一条 RoCE 链路，
            # 两台集群同时做操作会制造不必要的跨机同时突发（卡死本来就常成对出现）。
            for j, n in enumerate(list(BACKENDS)):
                if j:
                    await asyncio.sleep(DEEP_STAGGER)
                await _deep_one(n)
            _reconcile()
            await asyncio.sleep(_probe_interval())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[lb] 深度探针轮次异常（已忽略，继续下一轮）: "
                  f"{type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(HEALTH_INTERVAL)


def _probe_interval() -> float:
    """按「忙不忙」选探针频率：全死 → 中速盯恢复；全好且完全空闲 → 降频省电；
    其余（有流量、有台不健康）→ 高频。"""
    names = list(BACKENDS)
    if not any(bool(_health.get(n)) for n in names):
        return DEEP_INTERVAL_DEAD          # 全挂：中等频率盯谁先恢复
    now = time.time()
    idle = all((now - (_last_routed_at.get(n) or 0.0)) > IDLE_BACKOFF_AFTER
               for n in names)
    return DEEP_INTERVAL_IDLE if idle else DEEP_INTERVAL


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
        # keepalive_expiry=30：上游重启后，连接池里那些「写成功、读永久阻塞」的
        # 半开 socket 最多 30 秒就被回收。没有它时，表现为「直连 0.35 s、
        # 走代理 25 s 超时」—— 看着像集群挂了，其实是代理的连接池烂了。
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=32,
                            keepalive_expiry=30.0),
    )
    names = list(BACKENDS)
    results = await asyncio.gather(*(_probe(BACKENDS[n]) for n in names))
    for n, (ok, load) in zip(names, results):
        _shallow[n] = ok
        _sload[n] = load
    # 启动时先各验一发真实 completion 再放流量：没验过的集群一律不发
    await asyncio.gather(*(_deep_one(n) for n in names), return_exceptions=True)
    _reconcile()
    for n in names:
        print(f"[lb] {n} {BACKENDS[n]} -> "
              f"{'up' if _health.get(n) else 'down'}"
              f" (shallow={_shallow.get(n)} deep={_deep.get(n, {}).get('ok')}"
              f" {(_deep.get(n) or {}).get('ms')}ms)", flush=True)
    load_sticky()                      # 绑定表先读回来，重启后仍回到缓存所在的集群
    global _health_task, _deep_task
    _health_task = asyncio.create_task(_health_loop())
    _deep_task = asyncio.create_task(_deep_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    save_sticky(force=True)          # 停机前把绑定表落盘，下次启动仍回到同一台
    if _client:
        await _client.aclose()


# ── 模型名兼容 ──────────────────────────────────────────────────────────
def normalize_model(body: dict) -> None:
    m = body.get("model")
    if not isinstance(m, str) or m == REAL_MODEL:
        if not isinstance(m, str):
            body["model"] = REAL_MODEL
        return
    if m in LEGACY_MODELS:
        # 2026-10-07 旧方案（TensorFold EXL3 / DSV41）全部取消，现役仅 TP4 FP8。
        # 旧名不再静默映射，直接报错，逼着残留旧配置显式改模型名。
        raise HTTPException(
            status_code=400,
            detail=(f"model '{m}' 已下线：集群现仅提供 GLM-5.3-Flash-FP8"
                    "（TP4 环网），请改用该模型名"),
        )


# ── 转发 ────────────────────────────────────────────────────────────────
_DROP_REQ_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
_DROP_RES_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection"}


async def _forward(req: Request, cluster: str, body: bytes,
                   peek: bool = False) -> Response:
    url = BACKENDS[cluster] + req.url.path
    q = dict(req.query_params)
    headers = {k: v for k, v in req.headers.items() if k.lower() not in _DROP_REQ_HEADERS}
    _stats["routed"] += 1
    _stats["by_cluster"][cluster] = _stats["by_cluster"].get(cluster, 0) + 1
    _last_routed_at[cluster] = time.time()      # 探针据此判断是否完全空闲
    _inflight[cluster] = _inflight.get(cluster, 0) + 1

    rq = _client.build_request(req.method, url, content=body,
                               headers=headers, params=q or None)
    try:
        r = await _client.send(rq, stream=True)
    except Exception as e:
        _stats["errors"] += 1
        _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)   # 没发出去，不占用
        return JSONResponse(status_code=502, content={
            "error": {"message": f"upstream cluster {cluster} unreachable: {e}",
                      "type": "upstream_error", "cluster": cluster}})

    async def gen():
        # 卡死的引擎会「收下请求却永不吐字」。这一段要同时满足两件相反的要求：
        #  ① **记账要准**：一个不会结束的请求不能把 _inflight 永久占住（那正是泄漏的根源）；
        #  ② **别急着掐**：实测绝大多数卡死会在 30~50 s 内**自愈**，掐断等于把
        #     「客户端再等一下就能拿到完整结果」变成「客户端报错」。所以第一次静默
        #     只做「释放槽位 + 摘出流量」，**流继续留着等它自愈**；真的一直不回，
        #     到 STREAM_GIVEUP 才放弃（那时通常 memguard 已经把引擎重启了）。
        #
        # 用独立 watchdog 而不是 asyncio.wait_for 包 __anext__：后者超时会**取消**
        # 那个 await，可能把上游响应搞坏、吞掉 chunk。只读状态、不碰读循环。
        global _reaped
        it = r.aiter_raw()
        st = {"last": time.time(), "suspect": False, "released": False, "stop": False}

        async def watchdog() -> None:
            # 嵌套函数里赋值全局变量，**必须自己再声明一次 global** ——
            # 外层的 `global _reaped` 不会传进来，否则 `_reaped += 1` 会把它当局部
            # 变量而抛 UnboundLocalError，而这个 fire-and-forget 任务的异常会被静默
            # 吞掉：表现为「watchdog 悄悄死了，记账从此不准」（自检当场抓到的）。
            global _reaped
            while not st["stop"]:
                await asyncio.sleep(2)
                idle_for = time.time() - st["last"]
                if idle_for >= STREAM_GIVEUP:
                    print(f"[lb] {cluster} 回源流累计静默 {idle_for:.0f}s，放弃并关闭",
                          flush=True)
                    st["stop"] = True
                    # 光置标志没用：读循环正阻塞在上游那一行 read 上。
                    # 必须**主动关掉上游响应**，那个挂起的读才会被唤醒并结束，
                    # 否则这里会一直挂到客户端自己超时（实测 60s）。
                    try:
                        await r.aclose()
                    except Exception:
                        pass
                    return
                if idle_for >= STREAM_IDLE and not st["suspect"]:
                    st["suspect"] = True
                    st["released"] = True
                    _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)
                    _reaped += 1
                    print(f"[lb] {cluster} 回源流 {STREAM_IDLE:.0f}s 零字节 → 判疑似卡死："
                          f"已释放在途槽位、摘出流量，但**继续等**它自愈（累计 {_reaped}）",
                          flush=True)
                    _mark_dead(cluster, f"stream idle {STREAM_IDLE:.0f}s")
                elif idle_for < STREAM_IDLE and st["suspect"]:
                    st["suspect"] = False
                    st["released"] = False
                    _inflight[cluster] = _inflight.get(cluster, 0) + 1
                    print(f"[lb] {cluster} 回源流恢复出字，槽位已还回", flush=True)

        wd = asyncio.create_task(watchdog())
        try:
            async for chunk in it:
                st["last"] = time.time()
                yield chunk
        finally:
            st["stop"] = True
            wd.cancel()
            # 只有「没提前释放过」才在这里减，否则会重复减
            if not st["released"]:
                _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)
            save_sticky()                    # 去抖：2 秒内最多写一次
            try:
                await r.aclose()
            except Exception:
                pass

    out_headers = {k: v for k, v in r.headers.items()
                   if k.lower() not in _DROP_RES_HEADERS}
    if peek:
        # 验收用：回显本次命中的集群，让粘滞/均摊可被外部脚本直接观测
        out_headers["X-GLM-Cluster"] = cluster
    return StreamingResponse(
        gen(), status_code=r.status_code,
        media_type=r.headers.get("content-type"), headers=out_headers,
    )


async def _to_gateway(req: Request, body: bytes) -> Response:
    """Anthropic 协议 → OpenAI 协议转换。

    为什么必须自己做：8003 的 responses-gateway 只有 /v1/chat/completions、
    /v1/responses、/v1/models、/embeddings，**没有 /v1/messages**；
    而 GLM 的 TensorFold 后端同样只提供 OpenAI 侧端点。
    但 omp 的 local-anthropic provider 与 claude-code 的 ANTHROPIC_BASE_URL
    都指着 :8002 走 /v1/messages（过去靠 SGLang 原生支持）——
    不在这里补转换，那条链路会直接 404。
    """
    try:
        ab = json.loads(body or b"{}")
    except Exception:
        ab = {}
    if not isinstance(ab, dict):
        ab = {}
    _stats["anthropic"] = _stats.get("anthropic", 0) + 1

    want = ab.get("model", REAL_MODEL)
    if want in LEGACY_MODELS:
        # 2026-10-07 旧名随旧方案一并取消，不再静默映射（同 normalize_model）。
        raise HTTPException(
            status_code=400,
            detail=(f"model '{want}' 已下线：集群现仅提供 GLM-5.3-Flash-FP8"
                    "（TP4 环网），请改用该模型名"),
        )
    stream = bool(ab.get("stream"))

    # ── 请求：Anthropic → OpenAI ──────────────────────────────────────
    msgs = []
    sys_text = ab.get("system")
    if isinstance(sys_text, list):
        sys_text = "".join(p.get("text", "") for p in sys_text if isinstance(p, dict))
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    for m in ab.get("messages") or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            # 文本块拼成一条；工具调用块转成 OpenAI 的 tool_calls
            texts, calls = [], []
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    texts.append(p.get("text", ""))
                elif p.get("type") == "tool_use":
                    calls.append({"id": p.get("id", ""), "type": "function",
                                  "function": {"name": p.get("name", ""),
                                               "arguments": json.dumps(p.get("input", {}),
                                                                       ensure_ascii=False)}})
            if calls:
                msgs.append({"role": m.get("role", "assistant"),
                             "content": "".join(texts) or None, "tool_calls": calls})
            else:
                msgs.append({"role": m.get("role", "user"), "content": "".join(texts)})
        else:
            msgs.append({"role": m.get("role", "user"), "content": c or ""})
    # assistant 的 tool_use 转回 OpenAI 格式需要 role=tool 的回填，这里保持原样透传
    if ab.get("tools"):
        msgs_openai_tools = []
        for t in ab["tools"]:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            msgs_openai_tools.append({
                "type": "function",
                "function": {"name": t.get("name"),
                             "description": t.get("description", ""),
                             "parameters": t.get("input_schema") or {"type": "object"}}})
    else:
        msgs_openai_tools = None

    ob = {"model": want, "messages": msgs}
    if msgs_openai_tools:
        ob["tools"] = msgs_openai_tools
    for k_src, k_dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"),
                         ("top_p", "top_p"), ("top_k", "top_k")):
        if ab.get(k_src) is not None:
            ob[k_dst] = ab[k_src]
    if ab.get("stop_sequences"):
        ob["stop"] = ab["stop_sequences"]
    if ab.get("thinking", {}).get("type") == "disabled":
        ob["reasoning_effort"] = "none"
    if stream:
        ob["stream"] = True

    cluster, degraded = choose(session_key(ob))
    if degraded:
        _stats["bypassed"] += 1

    if stream:
        return await _forward_anthropic_stream(cluster, ob, want)
    return await _forward_anthropic_json(cluster, ob, want)


def _openai_msg_to_anthropic(msg: dict, model: str) -> dict:
    content = []
    rc = msg.get("reasoning_content")
    if rc:
        content.append({"type": "thinking", "thinking": rc})
    txt = msg.get("content")
    if isinstance(txt, list):                     # 少数后端会返回内容块数组
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    if txt:
        content.append({"type": "text", "text": txt})
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:
            args = {"_raw": f.get("arguments")}
        content.append({"type": "tool_use", "id": tc.get("id", ""),
                        "name": f.get("name", ""), "input": args})
    if not content:
        content.append({"type": "text", "text": ""})
    return {"id": "msg_" + hashlib.sha1(str(id(msg)).encode()).hexdigest()[:22],
            "type": "message", "role": "assistant", "model": model,
            "content": content,
            "stop_reason": "tool_use" if any(c["type"] == "tool_use" for c in content)
                           else "end_turn"}


async def _forward_anthropic_json(cluster: str, ob: dict, model: str) -> Response:
    url = BACKENDS[cluster] + "/v1/chat/completions"
    _stats["routed"] += 1
    _stats["by_cluster"][cluster] = _stats["by_cluster"].get(cluster, 0) + 1
    _inflight[cluster] = _inflight.get(cluster, 0) + 1
    try:
        try:
            r = await _client.post(url, content=json.dumps(ob, ensure_ascii=False).encode(),
                                   headers={"Content-Type": "application/json"})
        finally:
            _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)   # 非流式：拿到响应就释放
    except Exception as e:
        _stats["errors"] += 1
        return JSONResponse(status_code=502, content={
            "error": {"type": "upstream_error", "message": f"cluster {cluster}: {e}"}})
    if r.status_code >= 400:
        return JSONResponse(status_code=r.status_code, content=r.text)
    d = r.json()
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    u = d.get("usage") or {}
    out = _openai_msg_to_anthropic(msg, model)
    out["stop_reason"] = "max_tokens" if (d.get("choices") or [{}])[0].get("finish_reason") == "length" \
        else out["stop_reason"]
    out["usage"] = {"input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0)}
    return JSONResponse(out)


async def _forward_anthropic_stream(cluster: str, ob: dict, model: str) -> Response:
    """OpenAI SSE → Anthropic SSE 事件流。"""
    url = BACKENDS[cluster] + "/v1/chat/completions"
    _stats["routed"] += 1
    _stats["by_cluster"][cluster] = _stats["by_cluster"].get(cluster, 0) + 1
    _inflight[cluster] = _inflight.get(cluster, 0) + 1
    msg_id = "msg_" + hashlib.sha1(f"{cluster}{time.time()}".encode()).hexdigest()[:22]
    try:
        r = await _client.build_request("POST", url,
                                        content=json.dumps(ob, ensure_ascii=False).encode(),
                                        headers={"Content-Type": "application/json",
                                                 "Accept": "text/event-stream"})
        resp = await _client.send(r, stream=True)
    except Exception as e:
        _stats["errors"] += 1
        _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)   # 没发出去，不占用
        return JSONResponse(status_code=502, content={
            "error": {"type": "upstream_error", "message": f"cluster {cluster}: {e}"}})

    def sse(obj):
        return f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def gen():
        started = False
        in_tok = out_tok = 0
        try:
            head = sse({"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}})
            yield head.encode()
            started = True
            buf = ""
            # 同样的「无进展掐流」：上游收下请求却永不吐字时，这条流会永久挂着
            global _reaped
            ait = resp.aiter_text()
            while True:
                try:
                    raw = await asyncio.wait_for(ait.__anext__(), timeout=STREAM_IDLE)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    _reaped += 1
                    print(f"[lb] {cluster} anthropic 回源流 {STREAM_IDLE}s 零字节，"
                          f"判定卡死并掐掉（累计 {_reaped}）", flush=True)
                    _mark_dead(cluster, f"anthropic stream idle {STREAM_IDLE}s")
                    break
                buf += raw
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        yield sse({"type": "message_delta", "delta": {
                            "stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": out_tok}}).encode()
                        yield sse({"type": "message_stop"}).encode()
                        return
                    try:
                        d = json.loads(payload)
                    except Exception:
                        continue
                    u = d.get("usage")
                    if u:
                        in_tok = u.get("prompt_tokens", in_tok)
                        out_tok = u.get("completion_tokens", out_tok)
                    ch = (d.get("choices") or [{}])[0]
                    delta = ch.get("delta") or {}
                    for src, typ in (("reasoning_content", "thinking"), ("content", "text")):
                        v = delta.get(src)
                        if v:
                            blk = {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": typ + "_delta", typ: v}}
                            yield sse(blk).encode()
            yield sse({"type": "message_delta", "delta": {
                "stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": out_tok}}).encode()
            yield sse({"type": "message_stop"}).encode()
        finally:
            # 同步先放计数，再 await 关连接（理由同 _forward 的 gen）
            _inflight[cluster] = max(0, _inflight.get(cluster, 0) - 1)
            save_sticky()
            try:
                await resp.aclose()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


_ROUTE_PATHS = ["/v1/chat/completions", "/v1/completions", "/v1/responses",
                "/tokenize", "/detokenize", "/v1/tokenize", "/v1/detokenize"]

# Anthropic 协议路径：后端不支持，交给本机 8003 网关转换
_ANTHROPIC_PATHS = ["/v1/messages"]


async def _handle_anthropic(req: Request) -> Response:
    raw = await req.body()
    return await _to_gateway(req, raw)


for _p in _ANTHROPIC_PATHS:
    app.post(_p)(_handle_anthropic)


async def _handle(req: Request) -> Response:
    raw = await req.body()
    try:
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    normalize_model(body)
    key = session_key(body)

    forced = req.headers.get("x-glm-cluster", "").upper()
    if forced in BACKENDS:
        cluster, degraded = forced, False          # 验收/压测用：强制指定
    else:
        cluster, degraded = choose(key)
        if degraded:
            _stats["bypassed"] += 1
            print(f"[lb] bypass: {pick_cluster(key)} 不健康，本次走 {cluster}", flush=True)

    return await _forward(req, cluster, json.dumps(body, ensure_ascii=False).encode(),
                          peek=bool(req.headers.get("x-glm-peek")))


# 逐条注册（Starlette 的装饰器只接受单个路径，不能直接传列表）
for _p in _ROUTE_PATHS:
    app.post(_p)(_handle)


# ── 观测端点 ────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def models() -> Response:
    now = int(time.time())
    return JSONResponse({"object": "list", "data": [
        {"id": REAL_MODEL, "object": "model", "created": now, "owned_by": "mia-ai-lab"}]})


@app.get("/health")
async def health() -> Response:
    up = [n for n, v in _health.items() if v is True]
    return JSONResponse({
        "status": "ok" if len(up) == len(BACKENDS) else ("degraded" if up else "down"),
        "model": REAL_MODEL,
        "clusters": [{"name": n, "url": BACKENDS[n], "healthy": _health.get(n),
                      "shallow": _shallow.get(n),
                      "deep_ok": (_deep.get(n) or {}).get("ok"),
                      "deep_fails": (_deep.get(n) or {}).get("fails", 0),
                      "deep_ms": (_deep.get(n) or {}).get("ms"),
                      "deep_err": (_deep.get(n) or {}).get("err", ""),
                      "routed": _stats["by_cluster"].get(n, 0),
                      "inflight": _inflight.get(n, 0),
                      "running": (_sload.get(n) or {}).get("running", 0),
                      "waiting": (_sload.get(n) or {}).get("waiting", 0),
                      "paused": (_sload.get(n) or {}).get("paused"),
                      "filling": (_sload.get(n) or {}).get("filling"),
                      "pool_free_tokens": (_sload.get(n) or {}).get("pool_free_tokens"),
                      "iteration_s": (_sload.get(n) or {}).get("iteration_s"),
                      "lanes": (_sload.get(n) or {}).get("lanes"),
                      "bypassed_because_down": _bypassed.get(n, 0)}
                     for n in sorted(BACKENDS)],
        "routed_total": _stats["routed"],
        "reaped_streams": _reaped,
        "sticky_bindings": len(_sticky),
        "rebalanced": _rebound,
        "gate_fired": _gate_fired,
        "cooldown_blocked": _cooldown_blocked,
        "deep_probe": {"interval_s": DEEP_INTERVAL, "timeout_s": DEEP_TIMEOUT,
                       "stale_s": DEEP_STALE, "fail_tol": DEEP_FAIL_TOL,
                       "stream_idle_s": STREAM_IDLE, "stream_giveup_s": STREAM_GIVEUP},
        "saturation_gate": {"lanes_default": LANES_DEFAULT, "sat_oversub": SAT_OVERSUB,
                                         "move_cooldown_s": MOVE_COOLDOWN},
        "imbalance_gate_legacy": {"abs": IMBALANCE_ABS, "rel": IMBALANCE_REL},
    }, status_code=200 if up else 503)


@app.get("/lb/stats")
async def stats() -> Response:
    """验收用：粘滞是否生效、各集群分到多少、当前在途与排队。"""
    return JSONResponse({"stats": _stats, "health": _health,
                         "shallow": dict(_shallow),
                         "deep": {k: dict(v) for k, v in _deep.items()},
                         "reaped_streams": _reaped,
                         "loops_alive": {"deep": bool(_deep_task and not _deep_task.done()),
                                         "shallow": bool(_health_task and not _health_task.done())},
                         "probe_interval_now": _probe_interval(),
                         "inflight": dict(_inflight),
                         "server_load": {k: dict(v) for k, v in _sload.items()},
                         "sticky_bindings": len(_sticky), "rebalanced": _rebound,
                         "gate_fired": _gate_fired, "cooldown_blocked": _cooldown_blocked,
                         "saturation_gate": {"lanes_default": LANES_DEFAULT,
                                             "sat_oversub": SAT_OVERSUB,
                                             "move_cooldown_s": MOVE_COOLDOWN},
                         "imbalance_gate_legacy": {"abs": IMBALANCE_ABS, "rel": IMBALANCE_REL},
                         "backends": BACKENDS, "model": REAL_MODEL, "build": BUILD})


@app.get("/metrics")
async def metrics() -> Response:
    lines = [
        "# HELP glm53_lb_routed_total 请求总转发数",
        "# TYPE glm53_lb_routed_total counter",
        f"glm53_lb_routed_total {_stats['routed']}",
        "# HELP glm53_lb_bypassed_total 对端不健康导致的降级次数",
        "# TYPE glm53_lb_bypassed_total counter",
        f"glm53_lb_bypassed_total {_stats['bypassed']}",
        "# HELP glm53_lb_errors_total 上游不可达次数",
        "# TYPE glm53_lb_errors_total counter",
        f"glm53_lb_errors_total {_stats['errors']}",
    ]
    for n in sorted(BACKENDS):
        tag = re.sub(r"[^a-zA-Z0-9_]", "_", n)
        lines += [f"# TYPE glm53_lb_cluster_routed_total{{cluster=\"{tag}\"}} counter",
                  f"glm53_lb_cluster_routed_total{{cluster=\"{tag}\"}} "
                  f"{_stats['by_cluster'].get(n, 0)}",
                  f"# TYPE glm53_lb_cluster_inflight{{cluster=\"{tag}\"}} gauge",
                  f"glm53_lb_cluster_inflight{{cluster=\"{tag}\"}} {_inflight.get(n, 0)}",
                  f"# TYPE glm53_lb_cluster_waiting{{cluster=\"{tag}\"}} gauge",
                  f"glm53_lb_cluster_waiting{{cluster=\"{tag}\"}} "
                  f"{(_sload.get(n) or {}).get('waiting', 0)}",
                  f"# TYPE glm53_lb_cluster_running{{cluster=\"{tag}\"}} gauge",
                  f"glm53_lb_cluster_running{{cluster=\"{tag}\"}} "
                  f"{(_sload.get(n) or {}).get('running', 0)}",
                  f"glm53_lb_cluster_healthy{{cluster=\"{tag}\"}} "
                  f"{1 if _health.get(n) else 0}",
                  f"glm53_lb_cluster_deep_ok{{cluster=\"{tag}\"}} "
                  f"{1 if (_deep.get(n) or {}).get('ok') else 0}",
                  f"glm53_lb_cluster_shallow{{cluster=\"{tag}\"}} "
                  f"{1 if _shallow.get(n) else 0}",
                  f"glm53_lb_cluster_deep_ms{{cluster=\"{tag}\"}} "
                  f"{(_deep.get(n) or {}).get('ms') or 0}"]
    lines += ["# HELP glm53_lb_reaped_streams_total 回源流因零字节超时被掐掉的次数",
              "# TYPE glm53_lb_reaped_streams_total counter",
              f"glm53_lb_reaped_streams_total {_reaped}"]
    lines += ["# HELP glm53_lb_sticky_bindings 已记住的会话→集群绑定数（粘滞表大小）",
              "# TYPE glm53_lb_sticky_bindings gauge",
              f"glm53_lb_sticky_bindings {len(_sticky)}",
              "# HELP glm53_lb_moved_total 因饱和而真正换了台的次数",
              "# TYPE glm53_lb_moved_total counter",
              f"glm53_lb_moved_total {_rebound}",
              "# HELP glm53_lb_gate_fired_total 饱和判据触发次数（含被冷却期挡下的）",
              "# TYPE glm53_lb_gate_fired_total counter",
              f"glm53_lb_gate_fired_total {_gate_fired}",
              "# HELP glm53_lb_cooldown_blocked_total 因迟滞没搬成的次数",
              "# TYPE glm53_lb_cooldown_blocked_total counter",
              f"glm53_lb_cooldown_blocked_total {_cooldown_blocked}"]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


