import random


# Large prime used for Shamir's Secret Sharing
PRIME = 2**127 - 1


class ShamirSecretSharing:

    def __init__(self, threshold=2):

        if threshold < 2:
            raise ValueError(
                "Threshold must be at least 2."
            )

        self.threshold = threshold

    def _mod_inverse(self, value):

        return pow(
            value,
            PRIME - 2,
            PRIME
        )

    def split(self, secret, num_shares):

        if secret >= PRIME:
            raise ValueError(
                "Secret must be smaller than PRIME."
            )

        coefficients = [
            secret
        ]

        for _ in range(
                self.threshold - 1
        ):
            coefficients.append(
                random.randrange(1, PRIME)
            )

        shares = []

        for x in range(
                1,
                num_shares + 1
        ):

            y = 0

            for power, coefficient in enumerate(
                    coefficients
            ):

                y += (
                        coefficient *
                        pow(x, power, PRIME)
                )

            y %= PRIME

            shares.append(
                (x, y)
            )

        return shares

    def reconstruct(self, shares):

        if len(shares) < self.threshold:

            raise ValueError(
                "Not enough shares to reconstruct secret."
            )

        shares = shares[
                 :self.threshold
                 ]

        secret = 0

        for i, (x_i, y_i) in enumerate(
                shares
        ):

            numerator = 1
            denominator = 1

            for j, (x_j, _) in enumerate(
                    shares
            ):

                if i == j:
                    continue

                numerator *= -x_j
                denominator *= (
                        x_i - x_j
                )

            lagrange = (
                               numerator *
                               self._mod_inverse(
                                   denominator % PRIME
                               )
                       ) % PRIME

            secret += (
                    y_i * lagrange
            )

        return secret % PRIME