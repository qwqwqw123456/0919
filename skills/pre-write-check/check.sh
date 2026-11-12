#!/usr/bin/env bash
# pre-write-check —— 写操作前的 D/S/P/A 风险扫描（自 digital-employee-os v1.62 的
# pre-write-check.sh 移植，保留其"先切句 + 否定语境紧邻豁免"的判定口径）。
# 输入：stdin JSON {"text": "<操作描述>"}
# 退出码：0=低风险或否定豁免；1=命中高风险需确认；2=缺描述
set -u

PAYLOAD="$(cat)"
python3 - "$PAYLOAD" <<'PY'
import json
import re
import sys

raw = sys.argv[1].strip() if len(sys.argv) > 1 else ""
text = ""
if raw:
    try:
        text = str(json.loads(raw).get("text", "")).strip()
    except Exception:
        text = raw

if not text:
    print("用法：传入 {\"text\": \"<操作描述>\"}")
    sys.exit(2)

# 否定/禁止修饰：风险词紧邻这些词时表示"不执行该动作"，予以豁免
NEG = r'(?:不要|无需|无须|不用|不会|不能|不得|禁止|严禁|拒绝|避免|杜绝|未|没|勿|别|不|莫)'

# D/S/P/A 词表（与 safety-checklist 技能的四类对应，只增判定能力，不新增类别）
CATS = {
    "D-删除": r'删除|rm |del|remove|覆盖|overwrite|移动|mv |reset --hard|清空|truncate|drop |unlink',
    "S-发送": r'发送|发消息|发邮件|群发|发给|转发|发到|提交审批|发布|publish|post |sms|短信|outbox',
    "P-付费": r'付款|支付|下单|充值|兑换|订阅|开通|购买|purchase|pay |checkout',
    "A-权限/迁移": r'权限|授权|迁移|export|导出|批量下载|上传|chmod|chown|share|分享链接',
}

# 先按标点与转折词切句，避免"前半否定 + 后半肯定"被整句错误豁免
clauses = re.split(r'[，,；;。\n!！?？]|但是|不过|然而|可是|否则', text)

hit = []
for cat, pat in CATS.items():
    for cl in clauses:
        matched_one = False
        for m in re.finditer(pat, cl, re.I):
            pre = cl[max(0, m.start() - 6):m.start()]
            if re.search(NEG, pre):
                continue  # 否定修饰紧邻，该风险词被豁免
            matched_one = True
            break
        if matched_one:
            hit.append(cat)
            break

if hit:
    print("⛔ VERDICT=block 命中高风险类别: " + " ".join(hit))
    print("动作: 停手，先把对象/范围/可逆性摆给用户确认；确认后再执行，并在台账写“操作留痕”行（见 safety-checklist 技能）")
    sys.exit(0)  # 本运行时里非 0 会被当成工具故障，判定结果用文本传达

print("✅ VERDICT=pass 未命中 D/S/P/A 肯定动作，属低风险本地可逆操作（否定语境已豁免）")
print("💡 Scope creep 检查: 这个任务只读就能完成吗？能不写就不写")
sys.exit(0)
PY
