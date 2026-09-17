import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import secrets

from secret_sharing import (
    ShamirSecretSharing
)


def main():

    print("===================================")
    print("SECRET SHARING TEST")
    print("===================================")

    # Generate a random secret
    secret = secrets.randbelow(
        2**120
    )

    print("\nOriginal secret generated.")

    # 2-of-3 threshold
    sharing = ShamirSecretSharing(
        threshold=2
    )

    shares = sharing.split(
        secret,
        num_shares=3
    )

    print("Secret split into 3 shares.")

    # Use only two shares
    recovered_secret = (
        sharing.reconstruct(
            [
                shares[0],
                shares[2]
            ]
        )
    )

    print(
        "Secret reconstructed using 2 shares."
    )

    if recovered_secret == secret:

        print(
            "\nSecret recovery successful."
        )

        print(
            "SECRET SHARING TEST PASSED"
        )

    else:

        print(
            "\nSecret recovery failed."
        )

        print(
            "SECRET SHARING TEST FAILED"
        )

    print("===================================")


if __name__ == "__main__":
    main()