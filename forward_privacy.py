from key_manager import KeyManager
from crypto_utils import encrypt_data, decrypt_data
from pqc_utils import PQCKeyExchange


class ForwardPrivacyClient:

    def __init__(self, client_id, server_public_key):

        self.client_id = client_id

        # Create PQC helper
        self.pqc = PQCKeyExchange()

        # Client encapsulates using SERVER public key
        shared_secret, self.pqc_ciphertext = (
            self.pqc.encapsulate(server_public_key)
        )

        # Derive initial AES key from PQC shared secret
        initial_key = self.pqc.derive_aes_key(
            shared_secret
        )

        # Forward privacy key manager
        self.key_manager = KeyManager()

        # Start forward-privacy chain from PQC-derived key
        self.key_manager.current_key = initial_key

    def encrypt_update(self, plaintext):

        return encrypt_data(
            self.key_manager.get_current_key(),
            plaintext
        )

    def decrypt_update(self, encrypted_update):

        return decrypt_data(
            self.key_manager.get_current_key(),
            encrypted_update
        )

    def complete_round(self):

        self.key_manager.evolve_key()

    def get_current_key(self):

        return self.key_manager.get_current_key()

    def get_pqc_ciphertext(self):

        return self.pqc_ciphertext