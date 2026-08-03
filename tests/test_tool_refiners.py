import json

from app.tool_refiners import (
    DEFAULT_REFINERS,
    InsertIndexBoundsRefiner,
    TabAwareInspectRefiner,
)


def _inspect_result(payload: dict) -> str:
    return (
        "Document structure analysis for doc123:\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Link: https://docs.google.com/document/d/doc123/edit"
    )


ZERO_WITH_TABS = _inspect_result({
    "total_elements": 0,
    "tables": 0,
    "total_length": 0,
    "tabs": [{"title": "Terapia", "tab_id": "t.0"}],
})


def test_applies_on_zero_length_with_tabs():
    r = TabAwareInspectRefiner()
    assert r.applies("inspect_doc_structure", {"document_id": "doc123"}, ZERO_WITH_TABS)


def test_refined_arguments_add_first_tab_id():
    r = TabAwareInspectRefiner()
    args = {"document_id": "doc123", "detailed": False}
    refined = r.refined_arguments(args, ZERO_WITH_TABS)
    assert refined == {"document_id": "doc123", "detailed": False, "tab_id": "t.0"}


def test_refined_arguments_do_not_retrigger_applies():
    r = TabAwareInspectRefiner()
    refined = r.refined_arguments({"document_id": "doc123"}, ZERO_WITH_TABS)
    assert not r.applies("inspect_doc_structure", refined, ZERO_WITH_TABS)


def test_does_not_apply_to_other_tools():
    r = TabAwareInspectRefiner()
    assert not r.applies("get_doc_content", {"document_id": "doc123"}, ZERO_WITH_TABS)


def test_does_not_apply_when_tab_id_already_passed():
    r = TabAwareInspectRefiner()
    assert not r.applies(
        "inspect_doc_structure", {"document_id": "doc123", "tab_id": "t.0"}, ZERO_WITH_TABS
    )


def test_does_not_apply_on_real_length():
    r = TabAwareInspectRefiner()
    result = _inspect_result({"total_length": 55, "tabs": [{"title": "T", "tab_id": "t.0"}]})
    assert not r.applies("inspect_doc_structure", {"document_id": "doc123"}, result)


def test_does_not_apply_on_zero_length_without_tabs():
    r = TabAwareInspectRefiner()
    result = _inspect_result({"total_length": 0, "tabs": []})
    assert not r.applies("inspect_doc_structure", {"document_id": "doc123"}, result)


def test_does_not_apply_on_unparseable_result():
    r = TabAwareInspectRefiner()
    assert not r.applies("inspect_doc_structure", {"document_id": "doc123"}, "API error 404")


def test_first_tab_id_recurses_into_child_tabs():
    result = _inspect_result({
        "total_length": 0,
        "tabs": [{"title": "Parent", "child_tabs": [{"title": "Child", "tab_id": "t.child"}]}],
    })
    r = TabAwareInspectRefiner()
    refined = r.refined_arguments({"document_id": "doc123"}, result)
    assert refined["tab_id"] == "t.child"


def test_refiner_registered_by_default():
    assert any(isinstance(r, TabAwareInspectRefiner) for r in DEFAULT_REFINERS)
    assert any(isinstance(r, InsertIndexBoundsRefiner) for r in DEFAULT_REFINERS)


BOUND_ERROR = (
    "[Tool error] Error calling tool 'modify_doc_text': API error in modify_doc_text: "
    "<HttpError 400 ... returned \"Invalid requests[0].insertText: Index 65004 must be "
    'less than the end index of the referenced segment, 65004.">'
)


def test_bounds_applies_on_segment_end_error():
    r = InsertIndexBoundsRefiner()
    assert r.applies("modify_doc_text", {"document_id": "d", "start_index": 65004}, BOUND_ERROR)


def test_bounds_refined_arguments_use_api_bound_minus_one():
    r = InsertIndexBoundsRefiner()
    args = {"document_id": "d", "start_index": 65004, "text": "x"}
    refined = r.refined_arguments(args, BOUND_ERROR)
    assert refined == {"document_id": "d", "start_index": 65003, "text": "x"}


def test_bounds_refined_arguments_do_not_retrigger_applies():
    r = InsertIndexBoundsRefiner()
    refined = r.refined_arguments({"document_id": "d", "start_index": 65004}, BOUND_ERROR)
    assert not r.applies("modify_doc_text", refined, BOUND_ERROR)


def test_bounds_does_not_apply_to_replacement():
    r = InsertIndexBoundsRefiner()
    args = {"document_id": "d", "start_index": 65004, "end_index": 65010}
    assert not r.applies("modify_doc_text", args, BOUND_ERROR)


def test_bounds_does_not_apply_to_other_tools_or_errors():
    r = InsertIndexBoundsRefiner()
    assert not r.applies("insert_doc_elements", {"index": 5}, BOUND_ERROR)
    other = "[Tool error] API error: File not found: xyz."
    assert not r.applies("modify_doc_text", {"start_index": 5}, other)


def test_bounds_does_not_apply_when_bound_would_be_zero():
    r = InsertIndexBoundsRefiner()
    err = 'Index 1 must be less than the end index of the referenced segment, 1.'
    assert not r.applies("modify_doc_text", {"start_index": 1}, err)
