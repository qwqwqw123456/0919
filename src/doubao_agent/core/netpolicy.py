"""网络通信 / 模型等待 的策略层（Policy）。

把「等多久、重试几次、退避多长、轮询多密、连池多大」从各处的写死常量收进一份配置，
Agent 可以自己改这份配置来提速：`Policy.set()` 改完即生效（provider 每次调用读快照），
写回 agent.yaml 的 `policy:` 段，重启后依然生效。

生效边界（写在代码里也写给用的人）：
- 即时生效：超时、重试次数、退避、抖动、轮询间隔、重试状态码。
- 需重建连接：keepalive / 连接池大小 / http2（httpx 在客户端构造时固化）。

优先级：内置默认 < 预设 preset < agent.yaml `policy:` 段 < 环境变量
`AGENT_POLICY_<族>_<字段>`（例：AGENT_POLICY_MODEL_RETRIES=3）。
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable


@dataclass
class NetPolicy:
    """一次网络请求怎么等、失败怎么重来。"""

    connect_timeout: float = 3.0      # 建连超时（秒）
    read_timeout: float = 30.0        # 两次收包之间的超时
    write_timeout: float = 15.0       # 发送超时
    pool_timeout: float = 3.0         # 等连接池空闲槽的超时
    retries: int = 1                  # 同一次请求内允许的重试次数（不含首次）
    backoff_base: float = 0.15        # 退避基数：第 n 次重试等 base * 2^(n-1)
    backoff_max: float = 2.0          # 单次退避上限（秒）
    jitter: float = 0.25              # 退避抖动比例（0=不抖，避免同时重试撞车）
    retry_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    retry_on_timeout: bool = False    # 超时是否重试：默认否（端点卡死时重试只会把等待×N）
    keepalive: int = 20               # 连接保活池大小（需重建连接才生效）
    max_connections: int = 50         # 总连接上限（需重建连接才生效）
    http2: bool = False               # 需重建连接才生效


@dataclass
class ModelPolicy:
    """一次模型调用怎么等、超了怎么重来、异步任务怎么轮询。"""

    timeout: float = 120.0            # 整体超时（秒）
    first_byte: float = 20.0          # 首字节/首个响应头超时
    retries: int = 1                  # 模型级重试次数（不含首次）
    backoff_base: float = 0.2
    backoff_max: float = 1.5
    jitter: float = 0.25
    poll_interval: float = 1.0        # 等异步结果/远端任务的轮询起点（秒）
    poll_max: float = 30.0            # 轮询间隔上限（指数放大到顶）
    poll_timeout: float = 600.0       # 等一个异步结果的总时限，0=不限


DEFAULT = {"net": NetPolicy(), "model": ModelPolicy()}

# 预设：只给「改哪几档」，其余沿用默认。
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    # 追延迟：短超时 + 快重试 + 密轮询；代价是弱网下更容易失败
    "turbo": {
        "net": {"connect_timeout": 1.5, "read_timeout": 15.0, "retries": 2,
                "backoff_base": 0.08, "backoff_max": 0.6, "jitter": 0.4,
                "keepalive": 40},
        "model": {"timeout": 45.0, "first_byte": 8.0, "retries": 2,
                  "backoff_base": 0.1, "backoff_max": 0.8, "jitter": 0.4,
                  "poll_interval": 0.4, "poll_max": 5.0},
    },
    # 默认：够用，不太激进
    "balanced": {},
    # 追成功率：长超时 + 多重试 + 慢退避 + 稀疏轮询；代价是慢
    "careful": {
        "net": {"connect_timeout": 6.0, "read_timeout": 90.0, "retries": 3,
                "backoff_base": 0.6, "backoff_max": 8.0, "jitter": 0.2},
        "model": {"timeout": 300.0, "first_byte": 60.0, "retries": 3,
                  "backoff_base": 1.0, "backoff_max": 10.0,
                  "poll_interval": 2.0, "poll_max": 30.0},
    },
}

ENV_PREFIX = "AGENT_POLICY_"


class PolicyError(ValueError):
    """配置非法：宁可报错，不静默纠正成别的值。"""


def _coerce(default: Any, raw: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        value = int(raw)
        return value
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        if isinstance(raw, (list, tuple)):
            return tuple(int(x) for x in raw)
        return tuple(int(x) for x in str(raw).replace(";", ",").split(",") if str(x).strip())
    return str(raw)


class Policy:
    """一份可热改的策略表；线程安全，读的是不可变快照。"""

    def __init__(
        self,
        config_path: str | os.PathLike[str] | None = None,
        environ: dict[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.path = Path(config_path).expanduser() if config_path else None
        self._environ = dict(os.environ if environ is None else environ)
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._sources: dict[str, str] = {}
        self._mtime = -1.0
        self._version = 0
        self._net, self._model = replace(DEFAULT["net"]), replace(DEFAULT["model"])
        self._preset = ""
        self._latencies: dict[str, list[float]] = {"net": [], "model": []}
        self._fails: dict[str, int] = {"net": 0, "model": 0}
        for key, val in asdict(self._net).items():
            self._sources[f"net.{key}"] = "default"
        for key, val in asdict(self._model).items():
            self._sources[f"model.{key}"] = "default"
        self.reload(force=True)

    # ---------- 读 ----------
    def reload(self, force: bool = False) -> bool:
        """agent.yaml 变了就重读（按 mtime）；返回是否真的重载过。"""
        if self.path is None:
            return False
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        with self._lock:
            if not force and mtime == self._mtime:
                return False
            self._mtime = mtime
            self._net, self._model = replace(DEFAULT["net"]), replace(DEFAULT["model"])
            self._sources = {f"net.{k}": "default" for k in asdict(self._net)}
            self._sources.update({f"model.{k}": "default" for k in asdict(self._model)})
            self._preset = ""
            try:
                import yaml

                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 - 配置坏了不该让进程起不来
                self._version += 1
                return False
            block = dict((data.get("policy") or {}) if isinstance(data, dict) else {})
            preset = str(block.get("preset") or "").strip()
            if preset and preset not in PRESETS:
                raise PolicyError(f"未知 preset：{preset}（可用：{sorted(PRESETS)}）")
            if preset:
                self._apply_preset_dict(PRESETS[preset], source=f"preset:{preset}")
                self._preset = preset
            for fam in ("net", "model"):
                sub = block.get(fam) or {}
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        self._assign(fam, k, v, source="agent.yaml")
            self._apply_env()
            self._version += 1
            return True

    def _apply_preset_dict(self, payload: dict[str, dict[str, Any]], source: str) -> None:
        for fam, sub in payload.items():
            for k, v in sub.items():
                self._assign(fam, k, v, source=source)

    def _assign(self, fam: str, key: str, raw: Any, source: str) -> None:
        target = self._net if fam == "net" else self._model
        spec = {f.name: f for f in fields(target)}
        if key not in spec:
            raise PolicyError(f"未知策略项：{fam}.{key}")
        default = getattr(replace(DEFAULT[fam]), key)
        try:
            value = _coerce(default, raw)
        except (TypeError, ValueError) as exc:
            raise PolicyError(f"{fam}.{key} 的值不合法：{raw!r}") from exc
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise PolicyError(f"{fam}.{key} 不能为负：{value}")
        setattr(target, key, value)
        self._sources[f"{fam}.{key}"] = source

    def _apply_env(self) -> None:
        for env_key, raw in self._environ.items():
            if not env_key.startswith(ENV_PREFIX):
                continue
            rest = env_key[len(ENV_PREFIX):].lower()
            if "_" not in rest:
                continue
            fam, key = rest.split("_", 1)
            if fam not in ("net", "model"):
                continue
            self._assign(fam, key, raw, source=f"env:{env_key}")

    def snapshot(self) -> dict[str, Any]:
        """每次调用取一份快照：改配置不影响正在跑的那次请求。"""
        self.reload()
        with self._lock:
            return {
                "net": replace(self._net),
                "model": replace(self._model),
                "preset": self._preset,
                "version": self._version,
            }

    @property
    def net(self) -> NetPolicy:
        self.reload()
        with self._lock:
            return replace(self._net)

    @property
    def model(self) -> ModelPolicy:
        self.reload()
        with self._lock:
            return replace(self._model)

    @property
    def version(self) -> int:
        return self._version

    def explain(self) -> list[dict[str, Any]]:
        """每个字段当前值 + 它是从哪来的（default / preset / agent.yaml / env）。"""
        snap = self.snapshot()
        rows = []
        for fam in ("net", "model"):
            for k, v in asdict(snap[fam]).items():
                rows.append({"key": f"{fam}.{k}", "value": v,
                             "source": self._sources.get(f"{fam}.{k}", "?")})
        return rows

    # ---------- 写 ----------
    def set(self, key: str, raw: Any, persist: bool = True) -> dict[str, Any]:
        """改一项：形如 `model.retries`。persist=True 时写回 agent.yaml。"""
        if "." not in key:
            raise PolicyError("键要写成 族.字段，例如 model.retries")
        fam, name = key.split(".", 1)
        if fam not in ("net", "model"):
            raise PolicyError(f"未知策略族：{fam}")
        with self._lock:
            self._assign(fam, name, raw, source="set()")
            self._version += 1
            if persist and self.path is not None:
                self._persist()
            return {"key": key, "value": getattr(self._net if fam == "net" else self._model, name),
                    "version": self._version}

    def set_many(self, items: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        for k, v in items.items():
            self.set(k, v, persist=False)
        if persist and self.path is not None:
            self._persist()
        return {"changed": list(items), "version": self._version}

    def preset(self, name: str, persist: bool = True) -> dict[str, Any]:
        if name not in PRESETS:
            raise PolicyError(f"未知 preset：{name}（可用：{sorted(PRESETS)}）")
        with self._lock:
            self._net, self._model = replace(DEFAULT["net"]), replace(DEFAULT["model"])
            self._sources = {f"net.{k}": "default" for k in asdict(self._net)}
            self._sources.update({f"model.{k}": "default" for k in asdict(self._model)})
            self._apply_preset_dict(PRESETS[name], source=f"preset:{name}")
            self._preset = name
            self._apply_env()
            self._version += 1
            if persist and self.path is not None:
                self._persist()
        return {"preset": name, "version": self._version}

    def _persist(self) -> None:
        """只改 policy 段，其余配置原样写回。"""
        if self.path is None:
            return
        import yaml

        data = {}
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except OSError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        block: dict[str, Any] = {"preset": self._preset} if self._preset else {}
        for fam, obj in (("net", self._net), ("model", self._model)):
            sub = {}
            for f in fields(obj):
                key = f.name
                if self._sources.get(f"{fam}.{key}", "default") in ("default", f"preset:{self._preset}"):
                    continue
                sub[key] = getattr(obj, key)
            if sub:
                block[fam] = sub
        data["policy"] = block
        self.path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = -1.0

    # ---------- 观测与建议 ----------
    def record(self, fam: str, latency: float, ok: bool = True) -> None:
        with self._lock:
            bucket = self._latencies.setdefault(fam, [])
            bucket.append(max(0.0, float(latency)))
            del bucket[:-200]
            if not ok:
                self._fails[fam] = self._fails.get(fam, 0) + 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out = {}
            for fam, vals in self._latencies.items():
                if not vals:
                    out[fam] = {"n": 0, "p50": None, "p90": None, "fails": self._fails.get(fam, 0)}
                    continue
                s = sorted(vals)
                out[fam] = {"n": len(s), "p50": round(s[len(s) // 2], 3),
                            "p90": round(s[int(len(s) * 0.9) - 1], 3),
                            "max": round(s[-1], 3), "fails": self._fails.get(fam, 0)}
            return out

    def suggest(self) -> list[dict[str, Any]]:
        """按实测延迟给出「改哪几项能提速/更稳」的建议；只建议，不自动改。"""
        st = self.stats()
        tips = []
        m = st.get("model", {})
        if m.get("p50"):
            p50 = m["p50"]
            cur = self.model.first_byte
            if cur > max(2.0, p50 * 3):
                tips.append({"key": "model.first_byte", "to": round(max(2.0, p50 * 2), 1),
                             "why": f"p50 {p50}s，首字节上限 {cur}s 过于保守"})
            if m.get("p90") and self.model.timeout > max(10.0, m["p90"] * 6):
                tips.append({"key": "model.timeout", "to": round(max(10.0, m["p90"] * 4), 1),
                             "why": f"p90 {m['p90']}s，整体超时 {self.model.timeout}s 可收紧"})
        n = st.get("net", {})
        if n.get("fails") and self.net.backoff_base > 0.5:
            tips.append({"key": "net.backoff_base", "to": 0.2, "why": "失败多且退避过长，先降 base 再观察"})
        if n.get("fails", 0) >= 2 and self.net.retries < 2:
            tips.append({"key": "net.retries", "to": 2, "why": "已有 ≥2 次失败，单次请求重试 1→2 更稳"})
        return tips

    def sleep_backoff(self, fam: str, attempt: int) -> float:
        """第 attempt 次（1 起）重试前应睡多久：指数 + 上限 + 抖动。"""
        spec = self.net if fam == "net" else self.model
        raw = min(spec.backoff_max, spec.backoff_base * (2 ** max(0, attempt - 1)))
        if spec.jitter > 0:
            raw *= 1.0 + self._rng.uniform(-spec.jitter, spec.jitter)
        return max(0.0, raw)

    def httpx_timeout(self, fam: str = "net") -> Any:
        """给 httpx 用的超时对象（不引入 httpx 依赖时可当 dict 用）。"""
        spec = self.net if fam == "net" else self.model
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return {"connect": spec.connect_timeout}
        if fam == "model":
            return httpx.Timeout(self.model.timeout, connect=self.net.connect_timeout,
                                 read=self.model.first_byte, write=self.net.write_timeout,
                                 pool=self.net.pool_timeout)
        return httpx.Timeout(spec.read_timeout, connect=spec.connect_timeout,
                             read=spec.read_timeout, write=spec.write_timeout,
                             pool=spec.pool_timeout)


def load(config_path: str | os.PathLike[str] | None = None, environ: dict[str, str] | None = None) -> Policy:
    return Policy(config_path, environ)
