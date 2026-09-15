import sys
from pathlib import Path
import random

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from forward_privacy import ForwardPrivacyClient
from server import Server


def main():

    # -----------------------------
    # Configuration
    # -----------------------------

    number_of_clients = 5
    dropout_rate = 0.40

    client_ids = [
        f"Client_{i}"
        for i in range(1, number_of_clients + 1)
    ]

    # Create clients
    clients = {}

    for client_id in client_ids:

        clients[client_id] = (
            ForwardPrivacyClient(client_id)
        )

    server = Server()

    round_number = 1

    server.start_round(round_number)

    print("===================================")
    print("DROPOUT RESILIENCE SIMULATION")
    print("===================================")

    print(
        f"\nTotal clients: "
        f"{number_of_clients}"
    )

    print(
        f"Dropout rate: "
        f"{dropout_rate * 100:.0f}%"
    )

    # -----------------------------
    # Simulate dropout
    # -----------------------------

    dropped_clients = set(
        random.sample(
            client_ids,
            int(
                number_of_clients
                * dropout_rate
            )
        )
    )

    # -----------------------------
    # Clients participate
    # -----------------------------

    for client_id in client_ids:

        if client_id in dropped_clients:

            print(
                f"{client_id}: DROPPED"
            )

            continue

        model_update = (
            f"MODEL_UPDATE_FROM_{client_id}"
        ).encode()

        encrypted_update = (
            clients[client_id].encrypt_update(
                model_update
            )
        )

        server.receive_update(
            client_id,
            round_number,
            encrypted_update
        )

        print(
            f"{client_id}: UPDATE RECEIVED"
        )

    # -----------------------------
    # Analyze round
    # -----------------------------

    received_clients = (
        server.get_received_clients(
            round_number
        )
    )

    detected_drops = (
        server.get_dropped_clients(
            round_number,
            client_ids
        )
    )

    print("\n===================================")
    print("ROUND RESULT")
    print("===================================")

    print(
        f"Expected clients: "
        f"{len(client_ids)}"
    )

    print(
        f"Received updates: "
        f"{len(received_clients)}"
    )

    print(
        f"Dropped clients: "
        f"{len(detected_drops)}"
    )

    print(
        f"Dropped clients: "
        f"{detected_drops}"
    )


if __name__ == "__main__":
    main()