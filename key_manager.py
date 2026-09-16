import secrets
import hashlib


class KeyManager:

    def __init__(self):
        self.current_key = secrets.token_bytes(32)

    def evolve_key(self):
        old_key = self.current_key

        self.current_key = hashlib.sha256(
            b"FORWARD_PRIVACY_KEY_EVOLUTION" + old_key
        ).digest()

        return self.current_key

    def get_current_key(self):
        return self.current_key