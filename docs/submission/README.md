# Submission support and actual remaining requirements

The application and measured ensemble evidence are available. This folder does **not** constitute a completed academic submission. The user has chosen to complete the team-specific academic/administrative items manually. These are separate from the completed technical implementation. The actual requirement checklist remains `INCOMPLETE_TEAM_EVIDENCE_REQUIRED` until the team supplies the missing evidence; passing code tests does not establish authorship, novelty, presentation competence or permission to submit AI-assisted work.

- **technical-evidence-appendix.pdf:** a visually checked, three-page factual appendix with actual metrics and figures, embedded Times New Roman 11pt text and single leading. It is explicitly not the final student paper.
- **report-working-template.html:** an editable outline with Times New Roman 11pt/single-spacing print styles, chapter prompts and a suggested 40-page budget. Click the document to edit and use Download your edited outline to keep changes. Use the supplied institutional cover. This scaffold deliberately has missing team content; it is not padded to 40 pages or passed off as a finished report.
- **figures/:** independently shareable PNG/SVG figures, with source hashes. Development and held-out populations are kept separate.
- **primary-reading-list.md:** verified primary references and reading questions, not a completed literature-review argument.
- **contributions.csv:** blank real-work fields for three or four actual members. Suggested roles alone do not count as completed individual contributions.
- **ai-assistance-history.md:** assistance to verify and disclose using the institution's actual form; this is not an invented replacement form or permission record.
- **submission-manifest.json:** team-supplied file paths/declarations for the accepted proposal, final report, separate completed rubric, disclosure/permission, rehearsal and actual submission receipt.
- **readiness-result.json:** the honest current file-readiness result. The checker never certifies a grade or the authenticity of human declarations.

The brief asks for a group-compiled report around 40 pages and below 50, Times New Roman 11pt/single spacing, a cover, clear explanations/results, a project plan and actual member allocations, plus a separate completed rubric. It does not itself specify DOCX as mandatory; confirm accepted file types with the tutor. An editable DOCX path is optional in the checker. Provide the actual final paginated PDF for the mechanical page check and review its content/format manually.

The local native Word export stalled, so no unrendered DOCX is delivered. Its task-only hidden process was stopped. The PDF appendix was instead built directly with ReportLab, rendered through bundled Poppler and reviewed page by page. Unverified Word working files remain only in ignored runtime, not in the submission package.

Run the file checks from the repository root with the document Python environment (requires `pypdf` for actual report page/text checks):

```powershell
python autonomous-robot-navigation-rl/developer/commands/check_submission.py --output docs/submission/readiness-result.json
```

Exit 2 is expected while required team evidence is missing. If every file/declaration is supplied, the result still requires human review and is not an assignment-grade certificate. Do not submit this scaffold or a software-generated technical appendix as the team's own critical discussion. Resolve the actual Limited AI rules with the tutor and complete the prescribed disclosure.
