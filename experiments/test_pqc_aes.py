import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pqc_utils import PQCKeyExchange
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets


print("===================================")
print("ML-KEM + AES-GCM TEST")
print("===================================")

# Server creates ML-KEM key pair
server = PQCKeyExchange()

# Client uses server public key
client_pqc = PQCKeyExchange()
client_pqc.public_key = server.get_public_key()

# Client encapsulates
client_secret, ciphertext = client_pqc.encapsulate()

# Server decapsulates
server_secret = server.decapsulate(ciphertext)

# Check shared secret
if client_secret != server_secret:
    print("Shared secret mismatch.")
    print("PQC AES TEST FAILED")
    raise SystemExit

print("ML-KEM shared secret established.")

# Derive AES key
client_aes_key = client_pqc.derive_aes_key(client_secret)
server_aes_key = server.derive_aes_key(server_secret)

# Encrypt
plaintext = b"Federated Learning Model Update"

nonce = secrets.token_bytes(12)

aes = AESGCM(client_aes_key)
ciphertext_aes = nonce + aes.encrypt(
    nonce,
    plaintext,
    None
)

print("AES-GCM encryption successful.")

# Decrypt
received_nonce = ciphertext_aes[:12]
encrypted_data = ciphertext_aes[12:]

aes_server = AESGCM(server_aes_key)

decrypted = aes_server.decrypt(
    received_nonce,
    encrypted_data,
    None
)

print("AES-GCM decryption successful.")

if decrypted == plaintext:
    print("PQC + AES-GCM verification successful.")
    print("PQC AES TEST PASSED")
else:
    print("Decrypted data does not match.")
    print("PQC AES TEST FAILED")