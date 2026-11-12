"""text-stats：从 stdin 读取 {"text": ..., "top_k": ...}，输出 JSON 统计结果。"""

import json
import re
import sys
from collections import Counter


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        print(json.dumps({"error": "stdin 必须是 JSON"}, ensure_ascii=False))
        raise SystemExit(2)
    if not isinstance(payload, dict):
        print(json.dumps({"error": "输入必须是 JSON 对象"}, ensure_ascii=False))
        raise SystemExit(2)

    text = str(payload.get("text") or "")
    top_k = max(1, int(payload.get("top_k") or 10))

    words = [
        word
        for word in re.split(r"[^\w\u4e00-\u9fff]+", text.lower())
        if word and not word.isdigit()
    ]
    result = {
        "chars": len(text),
        "chars_no_space": len(re.sub(r"\s", "", text)),
        "cjk_chars": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "words": len(words),
        "lines": len(text.splitlines()),
        "top": Counter(words).most_common(top_k),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
