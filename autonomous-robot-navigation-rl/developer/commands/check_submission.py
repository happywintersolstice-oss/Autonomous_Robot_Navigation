"""Check files and declared readiness without inventing marks or human sign-offs.

Use the bundled document Python for full PDF checks. Passing file checks cannot
authenticate original authorship, contribution or tutor approval.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

APP = Path(__file__).resolve().parents[2]
REPO = APP.parent


def check(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    checks = []

    def exists(relative):
        return bool(relative and (base / relative).is_file())

    def add(name, passed, detail):
        checks.append(
            {
                "requirement": name,
                "file_or_declared_evidence_present": bool(passed),
                "detail": detail,
            }
        )

    members = data.get("members", [])
    valid_members = len(members) in (3, 4) and all(
        member.get("name")
        and member.get("registration_id")
        and member.get("actual_contribution")
        and exists(member.get("evidence_path"))
        for member in members
    )
    add(
        "Actual team and individual contributions",
        valid_members,
        "Requires3-4 real named members, IDs, actual work and existing evidence; role suggestions are not completed contributions",
    )
    proposal = data.get("approved_proposal", {})
    add(
        "Tutor accepted design",
        proposal.get("tutor_confirmed") and exists(proposal.get("path")),
        "A real accepted proposal/scope record is required",
    )
    report = data.get("final_report", {})
    report_files = exists(report.get("pdf_path"))
    report_ready = bool(report_files and report.get("team_authored_and_verified"))
    pages = None
    if report_files and report_ready:
        if report.get("docx_path"):
            if not exists(report["docx_path"]):
                report_ready = False
            else:
                with ZipFile(base / report["docx_path"]) as package:
                    body = ET.fromstring(package.read("word/document.xml"))
                    text = " ".join(
                        node.text or ""
                        for node in body.iter()
                        if node.tag.endswith("}t")
                    )
                    if "[TEAM:" in text or "TEAM CONTENT REQUIRED" in text:
                        report_ready = False
        try:
            from pypdf import PdfReader

            reader = PdfReader(base / report["pdf_path"])
            pages = len(reader.pages)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            if (
                "[TEAM:" in text
                or "TEAM CONTENT REQUIRED" in text
                or "not the final academic paper" in text
            ):
                report_ready = False
            report_ready = report_ready and 0 < pages < 50
        except ImportError:
            report_ready = False
        except Exception:
            # A corrupt/unreadable PDF is missing evidence, never a passing
            # declaration or an unhandled error in the remaining checks.
            report_ready = False
    add(
        "Completed student report and pagination",
        report_ready,
        f"Actual PDF pages: {pages}; brief targets around40 and requires below50. A formatting scaffold does not count. Team must review TNR11/single spacing, content and final layout. DOCX is optional unless the tutor requires that file type.",
    )
    rubric = data.get("completed_rubric", {})
    add(
        "Completed separate assessment rubric",
        rubric.get("team_verified") and exists(rubric.get("path")),
        "Use the actual completed rubric as its own file, not inside the report",
    )
    ai = data.get("ai_usage", {})
    ai_ready = (
        ai.get("actual_assistance_declared")
        and ai.get("tutor_permission_resolved")
        and exists(ai.get("official_disclosure_form_path"))
        and exists(ai.get("tutor_guidance_evidence_path"))
    )
    add(
        "Actual AI disclosure and permitted use",
        ai_ready,
        "Requires the prescribed form and resolved actual use; a newly checked box cannot retroactively prove permission",
    )
    presentation = data.get("presentation", {})
    add(
        "Individual demonstration and questions",
        presentation.get("each_member_can_explain_and_modify_own_work")
        and exists(presentation.get("rehearsal_evidence_path")),
        "Code tests do not certify a person's explanation or ability to modify their own work",
    )
    add(
        "Submission receipt",
        exists(data.get("classroom_submission", {}).get("receipt_path")),
        "This task does not submit to Google Classroom or contact the tutor",
    )
    ready = all(item["file_or_declared_evidence_present"] for item in checks)
    return {
        "status": (
            "DECLARED_EVIDENCE_PRESENT_REQUIRES_HUMAN_REVIEW"
            if ready
            else "INCOMPLETE_TEAM_EVIDENCE_REQUIRED"
        ),
        "all_rubric_requirements_certified": False,
        "grade_prediction": None,
        "checks": checks,
        "meaning": "File checks and user declarations only. Tutor assessment, authenticity, novelty and academic integrity cannot be certified by this program.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "docs/submission/submission-manifest.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.manifest)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"].startswith("DECLARED") else 2)


if __name__ == "__main__":
    main()
