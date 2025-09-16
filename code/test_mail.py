# Test Email

from utils import send_email_notification
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

send_email_notification(
    subject="✅ Regression Analysis",
    body="The analysis is complete, please check the files.",
)

# Run the command: python test_mail.py

# the email will be sent to the receiver email
