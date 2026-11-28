#!/usr/bin/env python3
"""策略层提速基准：同一份代码，只换 policy，量「等」的时间差。

三组实测，全部打本地 mock 端点，不连外网、不花钱：
1. 抖动端点：每 3 个请求 1 个 503，共 12 次调用 —— 看成功数与总耗时。
2. 卡死端点：响应永不到 —— 看多久才判死（120s 写死 vs 首字节超时）。
3. 异步任务轮询：任务 1.2s 完成 —— 看多久才发现完成。

基线 = policy=None（等价改造前写死的 120s / 不重试 / 2s 轮询）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from doubao_agent.core.a2a_client import A2AClient
from doubao_agent.core.netpolicy import Policy
from doubao_agent.providers.openai_compat import OpenAICompatProvider


class FlakyHandler(BaseHTTPRequestHandler):
    calls = 0
    every_fail = 3
    delay = 0.2
    hang = False
    counts = {"ok": 0, "503": 0}

    def do_POST(self):  # noqa: N802
        cls = type(self)
        cls.calls += 1
        self.rfile.read(int(self.headers.get("content-length", "0")))
        if cls.hang:
            time.sleep(90)
            return
        if cls.delay:
            time.sleep(cls.delay)
        if cls.calls % cls.every_fail == 0:
            cls.counts["503"] += 1
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"flaky"}')
            return
        cls.counts["ok"] += 1
        raw = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        return


def serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def workload(base_url: str, perf: Policy | None, n: int = 12):
    prov = OpenAICompatProvider(base_url=base_url, model="m", policy=perf)
    ok, fail, started = 0, 0, time.perf_counter()
    for _ in range(n):
        try:
            await prov.complete([{"role": "user", "content": "hi"}], [])
            ok += 1
        except Exception:  # noqa: BLE001
            fail += 1
    elapsed = time.perf_counter() - started
    await prov.aclose()
    return {"ok": ok, "fail": fail, "seconds": round(elapsed, 3),
            "avg_s": round(elapsed / n, 3)}


async def hang_probe(base_url: str, perf: Policy | None):
    prov = OpenAICompatProvider(base_url=base_url, model="m", policy=perf)
    started = time.perf_counter()
    try:
        await prov.complete([{"role": "user", "content": "hi"}], [])
        outcome = "returned"
    except Exception as exc:  # noqa: BLE001
        outcome = type(exc).__name__
    elapsed = time.perf_counter() - started
    await prov.aclose()
    return {"detect_after_s": round(elapsed, 2), "outcome": outcome}


async def hang_concurrent(base_url: str, perf: Policy | None, n: int = 3):
    """n 个请求同时卡住：看多久整批释放（并发场景下的真实吞吐影响）。"""
    started = time.perf_counter()
    await asyncio.gather(*[hang_probe(base_url, perf) for _ in range(n)])
    return {"release_after_s": round(time.perf_counter() - started, 2), "n": n}


async def poll_probe(perf: Policy | None, complete_at: float = 1.2):
    client = A2AClient("http://127.0.0.1:1", allow=["127.0.0.1"], policy=perf)
    started = time.monotonic()
    polls = {"n": 0}

    async def fake_get(task_id):
        polls["n"] += 1
        state = "completed" if time.monotonic() - started >= complete_at else "working"
        return {"status": {"state": state}}

    client.get = fake_get  # type: ignore[assignment]
    try:
        await client.wait("t1", max_wait=30.0)
    finally:
        await client.aclose()
    return {"detect_after_s": round(time.monotonic() - started, 2), "polls": polls["n"]}


def make(name: str, preset: str | None) -> Policy:
    p = Policy(None, environ={})
    if preset:
        p.preset(preset, persist=False)
    return p


def main() -> int:
    class Flaky(FlakyHandler):
        calls = 0
        counts = {"ok": 0, "503": 0}

    flaky_srv = serve(Flaky)
    base = f"http://127.0.0.1:{flaky_srv.server_address[1]}"
    cases = {
        "baseline(写死120s/不重试/2s轮询)": None,
        "balanced": make("balanced", "balanced"),
        "turbo": make("turbo", "turbo"),
    }
    out = {"workload": {}, "hang": {}, "poll": {}}
    for label, perf in cases.items():
        Flaky.calls = 0
        out["workload"][label] = asyncio.run(workload(base, perf))

    class Hang(FlakyHandler):
        calls = 0
        hang = True
        counts = {"ok": 0, "503": 0}

    hang_srv = serve(Hang)
    hang_base = f"http://127.0.0.1:{hang_srv.server_address[1]}"
    out["hang"]["baseline(首字节上限=整体120s)"] = asyncio.run(hang_probe(hang_base, None))
    out["hang"]["balanced(首字节20s)"] = asyncio.run(hang_probe(hang_base, cases["balanced"]))
    out["hang"]["turbo(首字节8s)"] = asyncio.run(hang_probe(hang_base, cases["turbo"]))

    for label, perf in cases.items():
        out.setdefault("hang_concurrent", {})[label] = asyncio.run(
            hang_concurrent(hang_base, perf, 3)
        )

    out["poll"]["baseline(2.0s)"] = asyncio.run(poll_probe(None))
    out["poll"]["balanced(1.0s)"] = asyncio.run(poll_probe(cases["balanced"]))
    out["poll"]["turbo(0.4s)"] = asyncio.run(poll_probe(cases["turbo"]))

    flaky_srv.shutdown()
    hang_srv.shutdown()

    # 退避总量：同样 4 次尝试，逐个 sleep 相加
    old = [0.2 * (1 + i) for i in range(3)]           # 旧：线性 n×0.2
    turbo = [cases["turbo"].sleep_backoff("model", i) for i in (1, 2, 3)]
    line = "退避总量（4 次尝试共等）：旧线性 %.2fs vs turbo %.2fs" % (sum(old), sum(turbo))
    out["backoff_note"] = {"old_linear_s": round(sum(old), 3),
                           "turbo_exp_s": round(sum(turbo), 3), "text": line}

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n=== 汇总 ===")
    for k, v in out["workload"].items():
        print(f"[workload] {k:34s} 成功 {v['ok']:2d}/12  失败 {v['fail']:2d}  总 {v['seconds']:6.2f}s")
    for k, v in out["hang"].items():
        print(f"[hang]     {k:34s} 判死 {v['detect_after_s']:6.2f}s  ({v['outcome']})")
    for k, v in out["poll"].items():
        print(f"[poll]     {k:34s} 发现完成 {v['detect_after_s']:5.2f}s  轮询 {v['polls']} 次")
    for k, v in out["hang_concurrent"].items():
        print(f"[hangx3]   {k:34s} 3 个并发卡死整批释放 {v['release_after_s']:6.2f}s")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
