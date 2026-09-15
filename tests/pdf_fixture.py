"""Shared minimal-PDF builder for tests (valid xref, parseable by pypdf)."""


def minimal_pdf_bytes(text: str = "Hello PDF world") -> bytes:
    """Build a tiny valid one-page PDF containing *text*."""
    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]"
            b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>"
        ),
        b"<</Length %d>>stream\n" % len(content) + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )
    return out
