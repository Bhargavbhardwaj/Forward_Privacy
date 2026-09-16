import secrets
import torch

from secret_sharing import ShamirSecretSharing


class DropoutMaskRecovery:

    def __init__(self, threshold=2):

        self.sharing = ShamirSecretSharing(
            threshold=threshold
        )

    def create_mask_seed(self):

        # Generate a cryptographically random integer
        # directly with Python instead of torch.randint().
        #
        # This avoids PyTorch's int64 limitation.
        return secrets.randbelow(
            2**120
        ) + 1

    def create_shares(
            self,
            seed,
            num_shares
    ):

        return self.sharing.split(
            seed,
            num_shares
        )

    def recover_seed(
            self,
            shares
    ):

        return self.sharing.reconstruct(
            shares
        )

    def generate_mask(
            self,
            seed,
            reference_update
    ):

        generator = torch.Generator()

        # PyTorch manual_seed accepts a signed 64-bit range.
        torch_seed = seed % (2**63 - 1)

        generator.manual_seed(
            torch_seed
        )

        mask = {}

        for name, parameter in (
                reference_update.items()
        ):

            mask[name] = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=parameter.dtype
            )

        return mask