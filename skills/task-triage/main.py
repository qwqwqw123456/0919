#!/usr/bin/env python3
"""task-triage —— 任务轻重分级（移植 v1.62 reasoning-engine 的委派分级 + 意图权重 + 预算三档）。

输入：stdin JSON {"task": "..."}
输出：分级、步骤预算、预估耗时/token、判定信号。
"""
import json
import re
import sys

# 意图权重（原引擎 WEIGHT 表；未识别按 1）
WEIGHT = {
    "逆向分析": 3, "信息调研": 2, "金融分析": 2, "预测分析": 1, "文档写作": 2,
    "图像创作": 2, "视频创作": 2, "演示文稿": 2, "合同法务": 1, "医疗咨询": 1,
    "表格数据": 1, "会议处理": 1, "定时任务": 1, "删除操作": 1, "对外发送": 1,
    "付费权益": 1, "权限迁移": 1, "体系工程": 1, "未识别": 1,
}

INTENT_HINTS = [
    ("逆向分析", r"逆向|反编译|脱壳|固件|APK|二进制|IDA|Ghidra"),
    ("信息调研", r"调研|研究|对比|盘点|梳理|查一下|搜一下"),
    ("金融分析", r"财报|股票|估值|毛利率|基金|汇率"),
    ("文档写作", r"写一份|文档|报告|总结|周报|复盘|方案"),
    ("演示文稿", r"PPT|幻灯片|演示|汇报材料"),
    ("图像创作", r"画一张|海报|配图|生图|设计图"),
    ("视频创作", r"视频|短片|成片|分镜"),
    ("表格数据", r"表格|Excel|CSV|统计|数据整理"),
    ("删除操作", r"删|清理|清空"),
    ("对外发送", r"发送|群发|发布|邮件"),
    ("付费权益", r"付费|充值|下单|购买|订阅"),
    ("权限迁移", r"权限|迁移|导出|授权"),
    ("体系工程", r"仓库|脚本|部署|重构|技能|运行时"),
]

BUDGET = {"L1-轻量": 5, "L2-标准": 12, "L3-委派": 20}


def intent_of(text: str) -> str:
    for name, pat in INTENT_HINTS:
        if re.search(pat, text, re.I):
            return name
    return "未识别"


def delegation_of(text: str):
    entities = len(re.findall(r"[一-鿿]{2,}(?:公司|产品|平台|模型|厂商|机构|人|仓库|系统)", text))
    entities += len(re.findall(r"(?:一|两|三|四|五|几|多)\s*(?:家|个|款|种|台|套|条)", text))
    multi = len(re.findall(r"(和|与|、|及|还有|多个|多路|并行|分别)", text))
    questions = len(re.findall(r"(是否|能不能|怎么|如何|为什么|对比|比较|哪个|几)", text))
    if entities >= 3 or multi >= 3 or questions >= 4:
        level = "L3-委派"
    elif entities >= 2 or multi + questions >= 2:
        level = "L2-标准"
    else:
        level = "L1-轻量"
    return level, {"实体": entities, "多主线词": multi, "疑问词": questions}


def main():
    raw = sys.stdin.read().strip()
    try:
        task = str(json.loads(raw).get("task", "")).strip()
    except Exception:
        task = raw
    if not task:
        print('用法：传入 {"task": "<任务原话>"}')
        return 2

    level, signals = delegation_of(task)
    intent = intent_of(task)
    weight = WEIGHT.get(intent, 1)
    if level == "L3-委派":
        weight = max(weight, 3)
    complexity = "重度" if weight >= 3 else ("标准" if weight == 2 else "轻量")
    # 分级与复杂度对齐：复杂度是重/标准时不允许预算停在轻量档
    floor = {"重度": "L3-委派", "标准": "L2-标准", "轻量": "L1-轻量"}[complexity]
    order = {"L1-轻量": 0, "L2-标准": 1, "L3-委派": 2}
    if order[floor] > order[level]:
        level = floor
    est = {
        "重度": (">15 分钟", "≈15k+ tokens（内容多时按 ≤3 片分块）"),
        "标准": ("5–15 分钟", "≈5–10k tokens"),
        "轻量": ("<5 分钟", "≈2–4k tokens"),
    }[complexity]

    print("分级: {} ｜ 意图: {}（权重 {}）｜ 复杂度: {}".format(level, intent, weight, complexity))
    print("步骤预算: ≤{} 步".format(BUDGET[level]))
    print("预估: 耗时 {} · token {}".format(est[0], est[1]))
    print("信号: " + " / ".join("{}={}".format(k, v) for k, v in signals.items()))
    if level == "L3-委派":
        print("提示: 命中 L3，先拆子任务再执行，单轮别硬扛")
    return 0


if __name__ == "__main__":
    sys.exit(main())
