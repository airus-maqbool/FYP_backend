from io import BytesIO
from xhtml2pdf import pisa


def generate_pdf_from_html(html_content: str) -> bytes:
    """
    Converts an HTML string into a PDF and returns the raw bytes.

    Uses xhtml2pdf — pure Python, 

    Install dependency:
        pip install xhtml2pdf

    Args:
        html_content: Plain HTML string (the same one sent to the frontend
                      for the email preview).

    Returns:
        PDF content as bytes, ready to be attached to an email.

    Raises:
        RuntimeError: If xhtml2pdf reports an error during conversion.
    """
    pdf_buffer = BytesIO()

    result = pisa.CreatePDF(
        src=html_content,
        dest=pdf_buffer,
        encoding="utf-8"
    )

    if result.err:
        raise RuntimeError(
            f"PDF generation failed with {result.err} error(s). "
            "Check your HTML for unsupported CSS or malformed tags."
        )
    print("pdf created")
    return pdf_buffer.getvalue()
