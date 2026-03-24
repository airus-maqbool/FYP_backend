import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def send_mom_email(
    recipient_email: str,
    subject: str,
    html_body: str,
    pdf_bytes: bytes,
    pdf_filename: str = "minutes_of_meeting.pdf"
) -> dict:
    """
    Sends an email with HTML body and a PDF attachment via SMTP.

    Args:
        recipient_email : The To address.
        subject         : Email subject line.
        html_body       : Plain HTML string received from the frontend.
        pdf_bytes       : PDF file content as bytes (generated from the same HTML).
        pdf_filename    : Name that will appear for the attachment.

    Returns:
        dict with status and message.
    """

    # ── Pull credentials from environment variables ──────────────────────────
    print("inside email sender 0")
    smtp_host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port     = int(os.getenv("SMTP_PORT", "465"))
    smtp_user     = os.getenv("SMTP_USER")          # your sending address
    smtp_password = os.getenv("SMTP_PASSWORD")      # app-password / SMTP password
    print("inside email sender 1")
    if not smtp_user or not smtp_password:
        raise ValueError(
            "SMTP credentials not set. "
            "Please set SMTP_USER and SMTP_PASSWORD environment variables."
        )
    print("inside email sender 2")
    # ── Build the email ───────────────────────────────────────────────────────
    msg = MIMEMultipart("mixed")
    msg["From"]    = smtp_user
    msg["To"]      = recipient_email
    msg["Subject"] = subject

    # HTML body (exactly what the frontend rendered)
    msg.attach(MIMEText(html_body, "html"))

    # PDF attachment
    pdf_part = MIMEBase("application", "pdf")
    pdf_part.set_payload(pdf_bytes)
    encoders.encode_base64(pdf_part)
    pdf_part.add_header(
        "Content-Disposition",
        "attachment",
        filename=pdf_filename
    )
    msg.attach(pdf_part)
    print("pdf attached")
    # port 465 k liye smtp_ssl, port 587 k liye starttls
    # ── Send via SMTP with STARTTLS ───────────────────────────────────────────
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient_email, msg.as_string())
    return {
        "status": "success",
        "message": f"Email sent successfully to {recipient_email}"
    }
