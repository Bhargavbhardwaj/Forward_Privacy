import hashlib
import secrets


class KeyManager:
    def __init__(self):
        self.current_key = secrets.token_bytes(32)

    def get_current_key(self):
        return self.current_key

    def evolve_key(self):
        old_key = self.current_key

        self.current_key = hashlib.sha256(
            b"FORWARD_PRIVACY_KEY_EVOLUTION" + old_key
        ).digest()

        # Remove our reference to the old key
        del old_key

        return self.current_key