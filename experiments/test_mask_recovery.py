import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from fl_model import SimpleNN
from secure_aggregation import SecureAggregator
from mask_recovery import MaskRecovery


def main():

    print("===================================")
    print("DROPOUT MASK RECOVERY TEST")
    print("===================================")

    client_ids = [
        "Client_1",
        "Client_2",
        "Client_3",
        "Client_4"
    ]

    # --------------------------------
    # Create client model updates
    # --------------------------------

    models = {}

    for client_id in client_ids:

        model = SimpleNN()

        models[client_id] = {
            name: parameter.clone()
            for name, parameter
            in model.state_dict().items()
        }

    print("\nClient updates created.")

    # --------------------------------
    # Create secure aggregator
    # --------------------------------

    secure_aggregator = SecureAggregator(
        client_ids
    )

    reference_model = models[
        client_ids[0]
    ]

    masks = secure_aggregator.create_masks(
        reference_model
    )

    print("Masks generated.")

    # --------------------------------
    # Mask all client updates
    # --------------------------------

    masked_updates = (
        secure_aggregator.create_masked_updates(
            models,
            masks
        )
    )

    print("All client updates masked.")

    # --------------------------------
    # Simulate dropout
    # --------------------------------

    dropped_client = "Client_4"

    surviving_ids = [
        "Client_1",
        "Client_2",
        "Client_3"
    ]

    surviving_masked_updates = {
        client_id: masked_updates[client_id]
        for client_id in surviving_ids
    }

    print(
        f"\nDropped client: {dropped_client}"
    )

    print("Surviving clients:")

    for client_id in surviving_ids:
        print(f"- {client_id}")

    # --------------------------------
    # Aggregate surviving masked updates
    # --------------------------------

    aggregate = {}

    first_client = surviving_ids[0]

    for name in surviving_masked_updates[
        first_client
    ]:

        aggregate[name] = torch.zeros_like(
            surviving_masked_updates[
                first_client
            ][name]
        )

    for client_id in surviving_ids:

        for name in aggregate:

            aggregate[name] += (
                surviving_masked_updates[
                    client_id
                ][name]
            )

    print(
        "\nSurviving masked updates aggregated."
    )

    # --------------------------------
    # Remove surviving masks
    # --------------------------------

    for client_id in surviving_ids:

        for name in aggregate:

            aggregate[name] -= (
                masks[client_id][name]
            )

    print(
        "Surviving masks removed."
    )

    # --------------------------------
    # Recover dropped client's mask
    # --------------------------------

    recovery = MaskRecovery(
        client_ids
    )

    dropped_mask = recovery.recover_mask(
        dropped_client,
        masks
    )

    print(
        f"Mask recovered for {dropped_client}."
    )

    # --------------------------------
    # IMPORTANT:
    #
    # The dropped client's update was
    # never included in the aggregate.
    #
    # Therefore its mask was also never
    # included and must NOT be removed.
    # --------------------------------

    recovered_aggregate = aggregate

    print(
        "Recovered aggregate reconstructed."
    )

    # --------------------------------
    # Calculate expected result
    # --------------------------------

    expected = {}

    for name in models[
        surviving_ids[0]
    ]:

        expected[name] = torch.zeros_like(
            models[surviving_ids[0]][name]
        )

    for client_id in surviving_ids:

        for name in expected:

            expected[name] += (
                models[client_id][name]
            )

    # --------------------------------
    # Verify
    # --------------------------------

    success = True

    for name in expected:

        if not torch.allclose(
                recovered_aggregate[name],
                expected[name],
                rtol=1e-4,
                atol=1e-4
        ):

            success = False

            difference = torch.max(
                torch.abs(
                    recovered_aggregate[name]
                    - expected[name]
                )
            ).item()

            print(
                f"Mismatch: {name} "
                f"(max difference: {difference})"
            )

    print("\n===================================")

    if success:
        print("MASK RECOVERY TEST PASSED")
    else:
        print("MASK RECOVERY TEST FAILED")

    print("===================================")


if __name__ == "__main__":
    main()