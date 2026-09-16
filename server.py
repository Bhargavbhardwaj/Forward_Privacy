from crypto_utils import decrypt_data
from pqc_utils import PQCKeyExchange


class Server:

    def __init__(self):

        self.received_updates = {}

        # Server owns ML-KEM key pair
        self.pqc = PQCKeyExchange()

        # Client receives this public key
        self.public_key = self.pqc.get_public_key()

        # Store derived client keys
        self.client_keys = {}

    def get_public_key(self):

        return self.public_key

    def establish_client_key(
            self,
            client_id,
            pqc_ciphertext
    ):

        # Server decapsulates client's ML-KEM ciphertext
        shared_secret = self.pqc.decapsulate(
            pqc_ciphertext
        )

        # Derive AES-256 key
        client_key = self.pqc.derive_aes_key(
            shared_secret
        )

        self.client_keys[client_id] = client_key

        return client_key

    def start_round(self, round_number):

        self.received_updates[round_number] = {}

    def receive_update(
            self,
            client_id,
            round_number,
            encrypted_update
    ):

        if round_number not in self.received_updates:

            self.received_updates[round_number] = {}

        self.received_updates[
            round_number
        ][client_id] = encrypted_update

    def get_received_clients(self, round_number):

        if round_number not in self.received_updates:

            return []

        return list(
            self.received_updates[
                round_number
            ].keys()
        )

    def get_dropped_clients(
            self,
            round_number,
            expected_clients
    ):

        received_clients = set(
            self.get_received_clients(
                round_number
            )
        )

        expected_clients = set(
            expected_clients
        )

        return sorted(
            expected_clients -
            received_clients
        )

    def decrypt_update(
            self,
            client_id,
            round_number
    ):

        encrypted_update = (
            self.received_updates[
                round_number
            ][client_id]
        )

        key = self.client_keys[client_id]

        return decrypt_data(
            key,
            encrypted_update
        )

    def evolve_client_key(self, client_id):

        old_key = self.client_keys[client_id]

        import hashlib

        self.client_keys[client_id] = hashlib.sha256(
            b"FORWARD_PRIVACY_KEY_EVOLUTION" + old_key
        ).digest()