import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from fl_model import SimpleNN
from secure_aggregation import SecureAggregator
from mask_recovery import DropoutMaskRecovery


def main():

    print("===================================")
    print("DROPOUT MASK RECOVERY INTEGRATION")
    print("===================================")

    client_ids = [
        "Client_1",
        "Client_2",
        "Client_3"
    ]

    dropped_client = "Client_3"

    surviving_clients = [
        "Client_1",
        "Client_2"
    ]

    # --------------------------------
    # Create model updates
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
    # Create mask recovery system
    # --------------------------------

    recovery = DropoutMaskRecovery(
        threshold=2
    )

    # --------------------------------
    # Generate mask seeds
    # --------------------------------

    seeds = {}

    for client_id in client_ids:

        seeds[client_id] = (
            recovery.create_mask_seed()
        )

    print("Mask seeds generated.")

    # --------------------------------
    # Generate masks
    # --------------------------------

    masks = {}

    reference_update = models[
        client_ids[0]
    ]

    for client_id in client_ids:

        masks[client_id] = (
            recovery.generate_mask(
                seeds[client_id],
                reference_update
            )
        )

    print("Masks generated.")

    # --------------------------------
    # Create recovery shares
    # --------------------------------

    shares = {}

    for client_id in client_ids:

        shares[client_id] = (
            recovery.create_shares(
                seeds[client_id],
                3
            )
        )

    print(
        "Recovery shares distributed."
    )

    # --------------------------------
    # Create masked updates
    # --------------------------------

    masked_updates = {}

    for client_id in client_ids:

        masked_updates[client_id] = {}

        for name in models[client_id]:

            masked_updates[
                client_id
            ][name] = (
                    models[client_id][name]
                    + masks[client_id][name]
            )

    print("Updates masked.")

    # --------------------------------
    # Simulate dropout
    # --------------------------------

    print(
        f"\nDropped client: {dropped_client}"
    )

    print("Surviving clients:")

    for client_id in surviving_clients:
        print(f"- {client_id}")

    # --------------------------------
    # Server receives only surviving
    # masked updates
    # --------------------------------

    received = {
        client_id: masked_updates[client_id]
        for client_id in surviving_clients
    }

    # --------------------------------
    # Aggregate received masked updates
    # --------------------------------

    aggregate = {}

    for name in reference_update:

        aggregate[name] = torch.zeros_like(
            reference_update[name]
        )

    for client_id in received:

        for name in aggregate:

            aggregate[name] += (
                received[client_id][name]
            )

    print(
        "\nSurviving masked updates aggregated."
    )

    # --------------------------------
    # Remove surviving masks
    # --------------------------------

    for client_id in surviving_clients:

        for name in aggregate:

            aggregate[name] -= (
                masks[client_id][name]
            )

    print(
        "Surviving masks removed."
    )

    # --------------------------------
    # Recover dropped client's seed
    #
    # Two surviving clients provide
    # their shares of Client_3's seed.
    # --------------------------------

    recovery_shares = [
        shares[dropped_client][0],
        shares[dropped_client][1]
    ]

    recovered_seed = (
        recovery.recover_seed(
            recovery_shares
        )
    )

    print(
        "Dropped client's mask seed recovered."
    )

    # --------------------------------
    # Regenerate dropped mask
    # --------------------------------

    recovered_mask = (
        recovery.generate_mask(
            recovered_seed,
            reference_update
        )
    )

    print(
        "Dropped client's mask regenerated."
    )

    # --------------------------------
    # Important:
    #
    # The dropped client's masked update
    # was NOT received by the server.
    #
    # Therefore its mask was never included
    # in the aggregate.
    #
    # We verify that the recovered mask matches
    # the original mask, demonstrating that
    # recovery is possible.
    # --------------------------------

    mask_recovered_correctly = True

    for name in masks[dropped_client]:

        if not torch.allclose(
                recovered_mask[name],
                masks[dropped_client][name],
                rtol=1e-4,
                atol=1e-4
        ):

            mask_recovered_correctly = False

            print(
                f"Mask mismatch: {name}"
            )

    # --------------------------------
    # Calculate expected aggregate
    # --------------------------------

    expected = {}

    for name in reference_update:

        expected[name] = torch.zeros_like(
            reference_update[name]
        )

    for client_id in surviving_clients:

        for name in expected:

            expected[name] += (
                models[client_id][name]
            )

    # --------------------------------
    # Verify aggregate
    # --------------------------------

    aggregate_correct = True

    for name in expected:

        if not torch.allclose(
                aggregate[name],
                expected[name],
                rtol=1e-4,
                atol=1e-4
        ):

            aggregate_correct = False

            difference = torch.max(
                torch.abs(
                    aggregate[name]
                    - expected[name]
                )
            ).item()

            print(
                f"Aggregate mismatch: {name} "
                f"(max difference: {difference})"
            )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n===================================")

    if (
            mask_recovered_correctly
            and aggregate_correct
    ):

        print(
            "DROPOUT MASK RECOVERY TEST PASSED"
        )

    else:

        print(
            "DROPOUT MASK RECOVERY TEST FAILED"
        )

    print("===================================")


if __name__ == "__main__":
    main()