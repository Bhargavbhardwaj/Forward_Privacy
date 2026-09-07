from crypto_utils import decrypt_data


class Server:

    def __init__(self):
        self.received_updates = {}

    def receive_update(
            self,
            client_id,
            round_number,
            encrypted_update
    ):
        """
        Store an encrypted update for a specific
        client and training round.
        """

        if client_id not in self.received_updates:
            self.received_updates[client_id] = {}

        self.received_updates[client_id][round_number] = (
            encrypted_update
        )

    def get_encrypted_update(
            self,
            client_id,
            round_number
    ):
        return self.received_updates[
            client_id
        ][round_number]

    def decrypt_update(
            self,
            client_id,
            round_number,
            key
    ):
        encrypted_update = self.get_encrypted_update(
            client_id,
            round_number
        )

        return decrypt_data(
            key,
            encrypted_update
        )