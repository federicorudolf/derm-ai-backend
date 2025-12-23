"""
Email service for sending transactional emails (password reset, etc.)
Uses SMTP with async support.
"""
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import asyncio
from dotenv import load_dotenv
import resend

load_dotenv()

logger = logging.getLogger(__name__)

resend.api_key = os.environ["RESEND_API_KEY"]
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

def create_password_reset_email(
    recipient_email: str,
    recipient_name: str,
    reset_token: str
):
    """
    Create password reset email with HTML template

    Args:
        recipient_email: Recipient's email address
        recipient_name: Recipient's name for personalization
        reset_token: The reset token (plain, not hashed)

    Returns:
        MIMEMultipart email message
    """
    reset_url = f"{FRONTEND_URL}/reset-password?token={reset_token}"

    # HTML email template
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
            .content {{ background-color: #f9fafb; padding: 30px; }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background-color: #4F46E5;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{ text-align: center; padding: 20px; color: #6b7280; font-size: 12px; }}
            .warning {{ color: #dc2626; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>DermAI - Restablecer Contraseña</h1>
            </div>
            <div class="content">
                <p>Hola {recipient_name},</p>
                <p>Recibimos una solicitud para restablecer la contraseña de tu cuenta DermAI.</p>
                <p>Haz clic en el siguiente botón para restablecer tu contraseña:</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Restablecer Contraseña</a>
                </p>
                <p>O copia y pega este enlace en tu navegador:</p>
                <p style="word-break: break-all; background-color: #e5e7eb; padding: 10px; border-radius: 5px;">
                    {reset_url}
                </p>
                <p class="warning">Este enlace expirará en 1 hora.</p>
                <p>Si no solicitaste restablecer tu contraseña, ignora este correo. Tu contraseña permanecerá sin cambios.</p>
                <p>Por seguridad, todas tus sesiones activas serán cerradas cuando cambies tu contraseña.</p>
            </div>
            <div class="footer">
                <p>Este es un correo automático, por favor no respondas a este mensaje.</p>
                <p>&copy; 2025 DermAI. Todos los derechos reservados.</p>
            </div>
        </div>
    </body>
    </html>
    """

    params: resend.Emails.SendParams = {
        "from": "dermAI <forgotpassword@dermai.com.ar>",
        "to": [recipient_email],
        "subject": "Restablecer tu contraseña de DermAI",
        "html": html_body
    }
    return params

def send_email_sync(params) -> bool:
    """
    Send email synchronously via Resend API

    Args:
        params: Resend email parameters dictionary

    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        email = resend.Emails.send(params)
        logger.info(f"Email sent successfully. ID: {email.get('id', 'unknown')}, To: {params['to']}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

async def send_email_async(message: MIMEMultipart) -> bool:
    """
    Send email asynchronously (runs sync function in thread pool)

    Args:
        message: MIMEMultipart email message

    Returns:
        True if email sent successfully, False otherwise
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, send_email_sync, message)

async def send_password_reset_email(
    recipient_email: str,
    recipient_name: str,
    reset_token: str
) -> bool:
    """
    High-level function to send password reset email

    Args:
        recipient_email: User's email address
        recipient_name: User's name
        reset_token: Plain reset token (not hashed)

    Returns:
        True if sent successfully, False otherwise
    """
    try:
        message = create_password_reset_email(recipient_email, recipient_name, reset_token)
        return await send_email_async(message)
    except Exception as e:
        logger.error(f"Error creating/sending password reset email: {e}")
        return False
