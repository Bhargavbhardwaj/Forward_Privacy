import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_data(key, plaintext):
    """
    Encrypt data using AES-256-GCM.

    Returns:
        nonce + ciphertext
    """

    nonce = os.urandom(12)

    aesgcm = AESGCM(key)

    ciphertext = aesgcm.encrypt(
        nonce,
        plaintext,
        None
    )

    return nonce + ciphertext


def decrypt_data(key, encrypted_data):
    """
    Decrypt AES-256-GCM encrypted data.

    Expects:
        nonce + ciphertext
    """

    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]

    aesgcm = AESGCM(key)

    plaintext = aesgcm.decrypt(
        nonce,
        ciphertext,
        None
    )

    return plaintext