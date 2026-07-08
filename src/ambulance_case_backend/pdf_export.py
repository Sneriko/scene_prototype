from __future__ import annotations

from textwrap import wrap

from .models import CaseOutput


def _escape_pdf_text_bytes(text: str) -> bytes:
    """Return a PDF literal string encoded for Helvetica/WinAnsi text drawing.

    PDF content streams for the built-in Helvetica font use single-byte font
    encodings. Encoding the stream as UTF-8 makes Swedish characters such as
    å, ä and ö render as mojibake in many PDF viewers. Windows-1252 maps those
    characters to the bytes expected by the font encoding, while still keeping
    the surrounding PDF drawing operators ASCII-compatible.
    """
    payload = text.encode("cp1252", errors="replace")
    return (
        payload.replace(b"\\", b"\\\\")
        .replace(b"(", b"\\(")
        .replace(b")", b"\\)")
    )


def _section_lines(title: str, body: str, *, width: int = 88) -> list[str]:
    lines = [title, ""]
    for paragraph in body.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(wrap(paragraph.strip(), width=width) or [""])
    return lines


def _build_pdf(title: str, subtitle: str, lines: list[str]) -> bytes:
    pages: list[list[str]] = []
    current: list[str] = []
    max_lines = 44
    for line in lines:
        current.append(line)
        if len(current) >= max_lines:
            pages.append(current)
            current = []
    if current or not pages:
        pages.append(current)

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{page_kids}] /Count {len(pages)} >>".encode("ascii"))

    for page_index, page_lines in enumerate(pages):
        page_object_id = 3 + page_index * 2
        stream_object_id = page_object_id + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
                f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> "
                f"/Contents {stream_object_id} 0 R >>"
            ).encode("ascii")
        )
        commands: list[bytes] = [
            b"BT /F2 20 Tf 56 742 Td 28 TL",
            b"(" + _escape_pdf_text_bytes(title) + b") Tj",
            b"T* /F1 10 Tf",
            b"(" + _escape_pdf_text_bytes(subtitle) + b") Tj",
            b"T* T* /F1 10 Tf 14 TL",
        ]
        for line in page_lines:
            commands.append(b"(" + _escape_pdf_text_bytes(line) + b") Tj")
            commands.append(b"T*")
        commands.append(b"ET")
        stream = b"\n".join(commands)
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")
    xref_position = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_position}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def treatment_pdf(case_output: CaseOutput) -> bytes:
    body_lines: list[str] = []
    for index, suggestion in enumerate(case_output.treatment_suggestions, start=1):
        body_lines.extend(
            _section_lines(
                f"{index}. {suggestion.title} ({suggestion.urgency.upper()})",
                suggestion.rationale,
            )
        )
        body_lines.append("")
    return _build_pdf(
        "Suggested treatment instructions",
        f"Case {case_output.case_id} · generated ambulance decision support",
        body_lines,
    )


def journal_pdf(case_output: CaseOutput) -> bytes:
    return _build_pdf(
        "Draft ambulance journal",
        f"Case {case_output.case_id} · draft for clinician review",
        _section_lines("Journal draft", case_output.drafted_journal),
    )
