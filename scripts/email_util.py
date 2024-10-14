import email
from bs4 import BeautifulSoup

def parse_email(raw_email):
    msg = email.message_from_string(raw_email)
    subject = msg['subject']
    from_ = msg['from']
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                body = BeautifulSoup(part.get_payload(decode=True), 'html.parser').text
    else:
        body = msg.get_payload()
    return subject, from_, body