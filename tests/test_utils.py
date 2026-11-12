from __future__ import annotations

from doubao_agent.types import Risk
from doubao_agent.utils import dumps, infer_risk, jsonable


def test_infer_risk_senior_keywords() -> None:
    assert infer_risk("sudo_install") is Risk.SENIOR
    assert infer_risk("cleanup", "rm_rf 清空目录") is Risk.SENIOR
    assert infer_risk("change_mode", "chmod 改权限") is Risk.SENIOR


def test_infer_risk_irreversible_keywords() -> None:
    assert infer_risk("delete_file") is Risk.IRREVERSIBLE
    assert infer_risk("send_message") is Risk.IRREVERSIBLE
    assert infer_risk("pay_order") is Risk.IRREVERSIBLE
    assert infer_risk("publish_article") is Risk.IRREVERSIBLE


def test_infer_risk_read_keywords() -> None:
    assert infer_risk("get_user") is Risk.READ
    assert infer_risk("list_records") is Risk.READ
    assert infer_risk("search_doc") is Risk.READ
    assert infer_risk("service", "查看运行状态") is Risk.READ


def test_infer_risk_default_reversible() -> None:
    assert infer_risk("update_note") is Risk.REVERSIBLE
    assert infer_risk("create_draft") is Risk.REVERSIBLE


def test_jsonable_recursively_converts_containers() -> None:
    payload = {"a": 1, "b": ["x", {"c": True}], "d": None}
    assert jsonable(payload) == {"a": 1, "b": ["x", {"c": True}], "d": None}


def test_jsonable_uses_model_dump_then_str() -> None:
    class Model:
        def model_dump(self) -> dict:
            return {"field": "v"}

    class Plain:
        def __str__(self) -> str:
            return "plain"

    assert jsonable(Model()) == {"field": "v"}
    assert jsonable(Plain()) == "plain"


def test_dumps_keeps_chinese_and_nested() -> None:
    text = dumps({"k": "向量检索", "n": [1, 2]})
    assert "向量检索" in text
    assert '"n": [1, 2]' in text
