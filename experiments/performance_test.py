import sys
from pathlib import Path
import time

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from key_manager import KeyManager
from crypto_utils import encrypt_data, decrypt_data


def measure_key_evolution():

    key_manager = KeyManager()

    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):
        key_manager.evolve_key()

    end = time.perf_counter()

    total_time = end - start

    average_time = (
            total_time / iterations
    )

    print("\nKey Evolution Performance")
    print("-------------------------")
    print(f"Iterations: {iterations}")
    print(
        f"Total time: {total_time:.6f} seconds"
    )
    print(
        f"Average time: "
        f"{average_time * 1000:.6f} ms"
    )


def measure_encryption():

    key_manager = KeyManager()

    key = key_manager.get_current_key()

    data = b"A" * 1024

    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):

        encrypt_data(
            key,
            data
        )

    end = time.perf_counter()

    total_time = end - start

    average_time = (
            total_time / iterations
    )

    print("\nEncryption Performance")
    print("----------------------")
    print(f"Data size: {len(data)} bytes")
    print(f"Iterations: {iterations}")
    print(
        f"Average time: "
        f"{average_time * 1000:.6f} ms"
    )


def measure_decryption():

    key_manager = KeyManager()

    key = key_manager.get_current_key()

    data = b"A" * 1024

    encrypted_data = encrypt_data(
        key,
        data
    )

    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):

        decrypt_data(
            key,
            encrypted_data
        )

    end = time.perf_counter()

    total_time = end - start

    average_time = (
            total_time / iterations
    )

    print("\nDecryption Performance")
    print("----------------------")
    print(f"Data size: {len(data)} bytes")
    print(f"Iterations: {iterations}")
    print(
        f"Average time: "
        f"{average_time * 1000:.6f} ms"
    )


if __name__ == "__main__":

    measure_key_evolution()
    measure_encryption()
    measure_decryption()