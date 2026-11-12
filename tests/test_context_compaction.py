from __future__ import annotations

from doubao_agent.core.agent import _compact, _conclusion_lines, _persist_digest, _trim_messages
from doubao_agent.core.memory import MemoryStore


def _history(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"u{index}"})
        messages.append(
            {
                "role": "assistant",
                "content": f"第{index}步检查完毕。结论：口径定为v{index}。其余细节略。",
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"t{index}", "content": "r"})
    return messages


def test_compact_keeps_user_points_and_newest_messages() -> None:
    messages = _history(10)

    trimmed, digest = _compact(messages, 8)

    assert len(trimmed) <= 8
    assert trimmed[0]["content"].startswith("sys")
    assert "早期要点" in trimmed[0]["content"]
    assert trimmed[1]["role"] == "user"
    assert digest, "被裁掉的助手结论应被抽出来"


def test_compact_extracts_assistant_conclusions_into_system_note() -> None:
    messages = _history(10)

    trimmed, digest = _compact(messages, 6)

    assert "【早期结论" in trimmed[0]["content"]
    assert any("结论" in line for line in digest)
    # 只保留含结论词的句子，不把整段助手回复塞进去
    assert all(len(line) <= 120 for line in digest)


def test_conclusion_lines_skips_tool_and_empty_messages() -> None:
    assert _conclusion_lines({"role": "tool", "content": "结论：随便"}) == []
    assert _conclusion_lines({"role": "assistant", "content": "   "}) == []
    assert _conclusion_lines(
        {"role": "assistant", "content": "先跑个脚本。结论：路径改为 scripts/。"}
    ) == ["结论：路径改为 scripts/"]


def test_conclusion_lines_falls_back_to_first_sentence() -> None:
    assert _conclusion_lines(
        {"role": "assistant", "content": "先摸底。再改代码。"}
    ) == ["先摸底"]


def test_trim_messages_wrapper_matches_compact() -> None:
    messages = _history(10)

    assert _trim_messages(messages, 8) == _compact(messages, 8)[0]


def test_compact_no_trim_when_within_limit() -> None:
    messages = _history(1)

    trimmed, digest = _compact(messages, 10)

    assert trimmed == messages
    assert digest == []


def test_persist_digest_writes_experience_memory(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "agent.db")

    memory_id = _persist_digest(memory, ["结论：只启用免费能力", "结论：凭证只显前4后4"])

    assert isinstance(memory_id, int)
    rows = memory.recall("早期结论沉淀", limit=5, kinds=("experience",))
    assert rows and "只启用免费能力" in rows[0]["text"]
    memory.close()


def test_persist_digest_ignores_empty_and_survives_failure() -> None:
    assert _persist_digest(None, ["结论：x"]) is None
    assert _persist_digest(object(), []) is None

    class Broken:
        def remember(self, *args, **kwargs):
            raise RuntimeError("db locked")

    assert _persist_digest(Broken(), ["结论：x"]) is None
