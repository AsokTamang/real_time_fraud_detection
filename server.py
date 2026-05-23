import uuid, base64
print(base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip('='))