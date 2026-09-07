from forward_privacy import ForwardPrivacyClient
from server import Server


def main():

    client = ForwardPrivacyClient(
        client_id="Client_1"
    )

    server = Server()

    print("===================================")
    print("FORWARD PRIVACY FL SIMULATION")
    print("===================================")

    for round_number in range(1, 4):

        print(f"\n--- Round {round_number} ---")

        # Simulated local model update
        model_update = (
            f"MODEL_UPDATE_FROM_CLIENT_1_ROUND_{round_number}"
        ).encode()

        # Save current key for experimental verification
        round_key = client.get_current_key()

        # Encrypt model update
        encrypted_update = client.encrypt_update(
            model_update
        )

        # Send encrypted update to server
        server.receive_update(
            client.client_id,
            round_number,
            encrypted_update
        )

        print("Client generated model update.")
        print("Model update encrypted.")
        print("Encrypted update sent to server.")

        # Verify server can decrypt current round
        decrypted_update = server.decrypt_update(
            client.client_id,
            round_number,
            round_key
        )

        print(
            "Server decrypted:",
            decrypted_update.decode()
        )

        # Move to next round key
        client.complete_round()

        print("Round completed.")
        print("Encryption key evolved.")


if __name__ == "__main__":
    main()