import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server import Server
from forward_privacy import ForwardPrivacyClient


print("===================================")
print("PQC + FORWARD PRIVACY TEST")
print("===================================")


# -----------------------------------
# Create server
# -----------------------------------

server = Server()

print("Server ML-KEM key pair generated.")


# -----------------------------------
# Create client
# -----------------------------------

client = ForwardPrivacyClient(
    "Client_1",
    server.get_public_key()
)

print("Client ML-KEM encapsulation successful.")


# -----------------------------------
# Establish shared key
# -----------------------------------

server.establish_client_key(
    "Client_1",
    client.get_pqc_ciphertext()
)

print("Server ML-KEM decapsulation successful.")


# -----------------------------------
# Round 1
# -----------------------------------

plaintext = b"Federated Learning Update - Round 1"

encrypted_update = client.encrypt_update(
    plaintext
)

server.start_round(1)

server.receive_update(
    "Client_1",
    1,
    encrypted_update
)

decrypted = server.decrypt_update(
    "Client_1",
    1
)

if decrypted == plaintext:

    print("Round 1 encryption/decryption successful.")

else:

    print("Round 1 FAILED.")

    raise SystemExit


# -----------------------------------
# Evolve both keys
# -----------------------------------

client.complete_round()

server.evolve_client_key(
    "Client_1"
)


# -----------------------------------
# Round 2
# -----------------------------------

plaintext2 = b"Federated Learning Update - Round 2"

encrypted_update2 = client.encrypt_update(
    plaintext2
)

server.start_round(2)

server.receive_update(
    "Client_1",
    2,
    encrypted_update2
)

decrypted2 = server.decrypt_update(
    "Client_1",
    2
)

if decrypted2 == plaintext2:

    print("Round 2 encryption/decryption successful.")

else:

    print("Round 2 FAILED.")

    raise SystemExit


# -----------------------------------
# Forward privacy test
# -----------------------------------

old_key = client.get_current_key()

client.complete_round()

new_key = client.get_current_key()

if old_key != new_key:

    print("Forward key evolution successful.")

else:

    print("Forward key evolution FAILED.")

    raise SystemExit


print()
print("PQC + Forward Privacy TEST PASSED")