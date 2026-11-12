from hostagent.contract import NODES, SOVEREIGN_SKILLS, Approval, Risk


def test_node_counts_match_manifest():
    local = [n for n in NODES.values() if n.delegate in ("data_agent", "delivery_agent")]
    external = [n for n in NODES.values() if n.delegate == "external_runtime"]
    assert len(NODES) == 21
    assert len(local) == 13
    assert len(external) == 8


def test_all_connectors_are_readonly():
    external = [n for n in NODES.values() if n.delegate == "external_runtime"]
    assert all(n.risk is Risk.READ and n.approval is Approval.NONE for n in external)


def test_send_file_supports_preauth():
    assert NODES["send_file"].approval is Approval.HUMAN_PREAUTH
    assert NODES["send_message"].approval is Approval.HUMAN
    assert NODES["create_calendar"].approval is Approval.HUMAN


def test_sovereign_skills_are_13():
    assert len(SOVEREIGN_SKILLS) == 13
    assert "weekly-report-generator" in SOVEREIGN_SKILLS
