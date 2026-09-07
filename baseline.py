from crypto_utils import encrypt_data, decrypt_data
import secrets


class StaticKeyClient:

    def __init__(self):
        # One key is used for every round
        self.key = secrets.token_bytes(32)

    def encrypt_update(self, model_update):
        return encrypt_data(
            self.key,
            model_update
        )

    def get_key(self):
        return self.key