from cryptography.hazmat.primitives.asymmetric import mlkem
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


class PQCKeyExchange:

    def __init__(self):
        # Used when this object acts as the server
        self.private_key = mlkem.MLKEM768PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_key(self):
        return self.public_key

    def encapsulate(self, public_key):
        """
        Client side.

        Use the server's ML-KEM public key to
        generate a shared secret and ciphertext.
        """

        shared_secret, ciphertext = public_key.encapsulate()

        return shared_secret, ciphertext

    def decapsulate(self, ciphertext):
        """
        Server side.

        Recover the shared secret using the
        server's private key.
        """

        shared_secret = self.private_key.decapsulate(
            ciphertext
        )

        return shared_secret

    def derive_aes_key(self, shared_secret):
        """
        Convert the ML-KEM shared secret into
        a 256-bit AES key using HKDF-SHA256.
        """

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"FL-PQC-AES-GCM"
        )

        return hkdf.derive(shared_secret)