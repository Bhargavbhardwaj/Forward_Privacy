import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from baseline import StaticKeyClient
from forward_privacy import ForwardPrivacyClient
from server import Server
from crypto_utils import decrypt_data


def test_static_key():

    print("\n==============================")
    print("STATIC KEY BASELINE")
    print("==============================")

    client = StaticKeyClient()
    server = Server()

    encrypted_updates = []

    for round_number in range(1, 4):

        model_update = (
            f"MODEL_UPDATE_ROUND_{round_number}"
        ).encode()

        encrypted_update = client.encrypt_update(
            model_update
        )

        encrypted_updates.append(
            encrypted_update
        )

        server.receive_update(
            "Client_Static",
            round_number,
            encrypted_update
        )

    # Attacker compromises the static key
    compromised_key = client.get_key()

    print("\nAttacker compromised the static key.")

    for round_number, encrypted_update in enumerate(
            encrypted_updates,
            start=1
    ):

        try:

            decrypted = decrypt_data(
                compromised_key,
                encrypted_update
            )

            print(
                f"Round {round_number}: "
                f"DECRYPTION SUCCESS"
            )

        except Exception:

            print(
                f"Round {round_number}: "
                f"DECRYPTION FAILED"
            )


def test_forward_private():

    print("\n==============================")
    print("FORWARD-PRIVATE SYSTEM")
    print("==============================")

    client = ForwardPrivacyClient(
        "Client_Forward_Private"
    )

    server = Server()

    encrypted_updates = []
    round_keys = []

    for round_number in range(1, 4):

        model_update = (
            f"MODEL_UPDATE_ROUND_{round_number}"
        ).encode()

        current_key = client.get_current_key()

        encrypted_update = client.encrypt_update(
            model_update
        )

        encrypted_updates.append(
            encrypted_update
        )

        round_keys.append(
            current_key
        )

        server.receive_update(
            "Client_Forward_Private",
            round_number,
            encrypted_update
        )

        # Key evolution
        client.complete_round()

    # Attacker compromises the current Round 3 key
    compromised_key = round_keys[2]

    print("\nAttacker compromised the Round 3 key.")

    for round_number, encrypted_update in enumerate(
            encrypted_updates,
            start=1
    ):

        try:

            decrypted = decrypt_data(
                compromised_key,
                encrypted_update
            )

            print(
                f"Round {round_number}: "
                f"DECRYPTION SUCCESS"
            )

        except Exception:

            print(
                f"Round {round_number}: "
                f"DECRYPTION FAILED"
            )


if __name__ == "__main__":

    test_static_key()

    test_forward_private()