import torch


class SecureAggregator:

    def __init__(self, client_ids):
        self.client_ids = client_ids

    def generate_mask_from_seed(self, seed, reference_update):
        """
        Deterministically generate a mask from a seed.
        The same seed produces the same mask.
        """

        generator = torch.Generator()

        # Keep seed inside PyTorch's supported range
        torch_seed = seed % (2**63 - 1)

        generator.manual_seed(torch_seed)

        mask = {}

        for name, parameter in reference_update.items():

            mask[name] = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=parameter.dtype
            )

        return mask

    def create_masked_update(
            self,
            client_update,
            mask
    ):
        """
        Add mask to a client update.
        """

        masked_update = {}

        for name in client_update:

            masked_update[name] = (
                    client_update[name]
                    + mask[name]
            )

        return masked_update

    def remove_mask(
            self,
            masked_update,
            mask
    ):
        """
        Remove a mask from an update.
        """

        unmasked_update = {}

        for name in masked_update:

            unmasked_update[name] = (
                    masked_update[name]
                    - mask[name]
            )

        return unmasked_update

    def aggregate_masked_updates(
            self,
            masked_updates
    ):
        """
        Sum masked updates.
        """

        if not masked_updates:
            raise ValueError(
                "No masked updates available."
            )

        first_client = next(
            iter(masked_updates)
        )

        aggregate = {}

        for name, parameter in masked_updates[
            first_client
        ].items():

            aggregate[name] = torch.zeros_like(
                parameter
            )

        for client_id in masked_updates:

            for name in aggregate:

                aggregate[name] += (
                    masked_updates[
                        client_id
                    ][name]
                )

        return aggregate