import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pqc_utils import PQCKeyExchange


print("===================================")
print("PQC ML-KEM-768 TEST")
print("===================================")

# Server generates ML-KEM key pair
server = PQCKeyExchange()

public_key = server.get_public_key()

print("ML-KEM-768 key pair generated.")

# Client encapsulates a shared secret using server public key
client_secret, ciphertext = server.encapsulate(public_key)

print("Encapsulation successful.")
print("Ciphertext generated.")

# Server decapsulates
server_secret = server.decapsulate(ciphertext)

print("Decapsulation successful.")

# Verify both parties obtained same secret
if client_secret == server_secret:
    print("Shared secret verification successful.")
    print("PQC TEST PASSED")
else:
    print("Shared secrets do not match.")
    print("PQC TEST FAILED")