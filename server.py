from crypto_utils import decrypt_data


class Server:

    def __init__(self):
        self.received_updates = {}

    def start_round(self, round_number):
        """
        Initialize storage for a new FL round.
        """

        self.received_updates[round_number] = {}

    def receive_update(
            self,
            client_id,
            round_number,
            encrypted_update
    ):
        """
        Store an encrypted client update.
        """

        if round_number not in self.received_updates:
            self.received_updates[round_number] = {}

        self.received_updates[
            round_number
        ][client_id] = encrypted_update

    def get_received_clients(self, round_number):
        """
        Return clients whose updates were received.
        """

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
        """
        Determine which expected clients
        did not send an update.
        """

        received_clients = set(
            self.get_received_clients(
                round_number
            )
        )

        expected_clients = set(
            expected_clients
        )

        return sorted(
            expected_clients - received_clients
        )

    def decrypt_update(
            self,
            client_id,
            round_number,
            key
    ):
        """
        Decrypt a client's encrypted update.
        """

        encrypted_update = (
            self.received_updates[
                round_number
            ][client_id]
        )

        return decrypt_data(
            key,
            encrypted_update
        )