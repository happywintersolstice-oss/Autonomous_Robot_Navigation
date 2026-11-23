"""Readiness must never invent student contributions or certify a grade."""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile
from robot_navigation.core import ROOT


def checker():
    path = ROOT / "developer/commands/check_submission.py"
    spec = importlib.util.spec_from_file_location("submission_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_team_evidence_is_reported_honestly(tmp_path):
    # Keep this a temporary fixture so the team's later completion does not
    # make a valid submission fail the software regression suite.
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"members": []}))
    result = checker().check(path)
    assert result["status"] == "INCOMPLETE_TEAM_EVIDENCE_REQUIRED"
    assert result["all_rubric_requirements_certified"] is False
    assert result["grade_prediction"] is None
    assert len(result["checks"]) == 7


def test_report_scaffold_is_not_treated_as_a_finished_report(tmp_path):
    with ZipFile(tmp_path / "report.docx", "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>[TEAM: write your own discussion]</w:t></w:r></w:p></w:body></w:document>',
        )
    (tmp_path / "report.pdf").write_bytes(b"not-a-submission")
    data = {
        "final_report": {
            "docx_path": "report.docx",
            "pdf_path": "report.pdf",
            "team_authored_and_verified": False,
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    # No PDF parsing is necessary to reject an explicitly unfinished scaffold.
    module = checker()
    result = module.check(tmp_path / "manifest.json")
    assert result["status"] == "INCOMPLETE_TEAM_EVIDENCE_REQUIRED"
    assert not next(
        row
        for row in result["checks"]
        if row["requirement"].startswith("Completed student")
    )["file_or_declared_evidence_present"]


def test_empty_roles_are_not_individual_contribution_evidence(tmp_path):
    data = {
        "members": [
            {"name": "", "registration_id": "", "actual_contribution": ""}
            for _ in range(4)
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    result = checker().check(path)
    assert not result["checks"][0]["file_or_declared_evidence_present"]
