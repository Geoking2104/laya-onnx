from laya_onnx.ultrafast.ops import OPS
from laya_onnx.ultrafast.policy import questions
from laya_onnx.ultrafast.snapshot import Element, PageSnapshot
from laya_onnx.ultrafast.browser import MockBrowser

def test_questions_cover_ops_and_targets():
    snap = PageSnapshot(url="https://example.com", title="Flights", elements=[Element(0, "button", "One way"), Element(1, "textbox", "Where from?", editable=True)], text="Flights")
    q = questions(snap, "Zurich to London one way")
    assert set(q["op"]["criteria"]) == set(OPS)
    assert q["ready"]["type"] == "noul"
    assert "One way" in q["target"]["criteria"][0]

def test_mock_browser_types():
    snap = PageSnapshot(url="https://example.com", elements=[Element(0, "textbox", "Where from?", editable=True)])
    b = MockBrowser(snap)
    b.type_text(0, "Zurich")
    assert b.snap.elements[0].value == "Zurich"
