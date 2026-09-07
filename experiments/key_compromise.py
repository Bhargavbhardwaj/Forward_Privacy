import sys
from pathlib import Path

# Allow importing modules from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from key_manager import KeyManager
from crypto_utils import encrypt_data, decrypt_data


def main():

    key_manager = KeyManager()

    encrypted_updates = []
    round_keys = []

    # -----------------------------
    # Generate encrypted updates
    # -----------------------------

    for round_number in range(1, 4):

        current_key = key_manager.get_current_key()

        round_keys.append(current_key)

        model_update = (
            f"CLIENT_MODEL_UPDATE_ROUND_{round_number}"
        ).encode()

        encrypted_update = encrypt_data(
            current_key,
            model_update
        )

        encrypted_updates.append(encrypted_update)

        print(f"Round {round_number} completed.")

        # Move to next key
        key_manager.evolve_key()

    # -----------------------------
    # Simulate key compromise
    # -----------------------------

    compromised_key = round_keys[2]

    print("\n==============================")
    print("SIMULATING KEY COMPROMISE")
    print("==============================")

    print("\nAttacker obtained Round 3 key.")

    # -----------------------------
    # Attempt to decrypt all rounds
    # -----------------------------

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

            print(
                f"Recovered data: {decrypted.decode()}"
            )

        except Exception:

            print(
                f"Round {round_number}: "
                f"DECRYPTION FAILED"
            )


if __name__ == "__main__":
    main()