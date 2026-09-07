from key_manager import KeyManager
from crypto_utils import encrypt_data, decrypt_data


class ForwardPrivacyClient:

    def __init__(self, client_id):
        self.client_id = client_id
        self.key_manager = KeyManager()

    def encrypt_update(self, model_update):
        """
        Encrypt a client's model update using
        the current round key.
        """

        current_key = self.key_manager.get_current_key()

        encrypted_update = encrypt_data(
            current_key,
            model_update
        )

        return encrypted_update

    def complete_round(self):
        """
        Evolve the key after a training round.
        """

        self.key_manager.evolve_key()

    def get_current_key(self):
        """
        Returns the current key.
        Used only for controlled experiments.
        """

        return self.key_manager.get_current_key()