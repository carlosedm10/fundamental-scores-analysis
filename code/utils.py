import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import mimetypes
from dotenv import load_dotenv

load_dotenv()


def send_email_notification(
    subject: str,
    body: str,
    image_path: str | list[str] | None = None,
):
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if not all([sender_email, sender_password, receiver_email]):
        print("Email configuration incomplete. Skipping email notification.")
        return

    # Create message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Add body
    msg.attach(MIMEText(body, "plain"))

    # Attach images if provided and exist
    if image_path:
        # Handle both single path and list of paths
        image_paths = [image_path] if isinstance(image_path, str) else image_path

        for path in image_paths:
            if not os.path.exists(path):
                print(f"Skipping attachment: {path} does not exist.")
                continue

            ctype, encoding = mimetypes.guess_type(path)
            # if ctype is None or not ctype.startswith("image/"):
            #     print(f"Skipping attachment: {path} is not a valid image.")
            #     continue

            maintype, subtype = ctype.split("/", 1)
            with open(path, "rb") as f:
                img = MIMEImage(f.read(), _subtype=subtype)
                img.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(path),
                )
                msg.attach(img)

    # Send email
    try:
        with smtplib.SMTP("smtp.zoho.eu", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email notification sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
