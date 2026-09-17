import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from fl_model import SimpleNN
from secure_aggregation import SecureAggregator


def main():

    print("===================================")
    print("SECURE AGGREGATION TEST")
    print("===================================")

    client_ids = [
        "Client_1",
        "Client_2",
        "Client_3"
    ]

    # Create model updates for each client
    models = {}

    for client_id in client_ids:

        model = SimpleNN()

        models[client_id] = {
            name: parameter.clone()
            for name, parameter in model.state_dict().items()
        }

    print("\nClient updates created.")

    # Create secure aggregator
    aggregator = SecureAggregator(client_ids)

    reference_model = models[client_ids[0]]

    # Generate random masks
    masks = aggregator.create_masks(reference_model)

    print("Masks generated.")

    # Mask client updates
    masked_updates = aggregator.create_masked_updates(
        models,
        masks
    )

    print("Updates masked.")

    # Aggregate masked updates
    secure_sum = aggregator.aggregate_masked_updates(
        masked_updates,
        masks
    )

    print("Secure aggregation completed.")

    # Calculate expected ordinary sum
    expected_sum = {}

    for name in reference_model:

        expected_sum[name] = (
                models["Client_1"][name]
                + models["Client_2"][name]
                + models["Client_3"][name]
        )

    # Compare results
    aggregation_correct = True

    for name in expected_sum:

        if not torch.allclose(
                secure_sum[name],
                expected_sum[name],
                rtol=1e-4,
                atol=1e-4
        ):

            aggregation_correct = False

            difference = torch.max(
                torch.abs(
                    secure_sum[name] -
                    expected_sum[name]
                )
            ).item()

            print(
                f"Mismatch: {name} "
                f"(max difference: {difference})"
            )

    print("\n===================================")

    if aggregation_correct:

        print("SECURE AGGREGATION TEST PASSED")

    else:

        print("SECURE AGGREGATION TEST FAILED")

    print("===================================")


if __name__ == "__main__":
    main()