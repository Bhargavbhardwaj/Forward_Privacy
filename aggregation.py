import torch


def fed_avg(client_updates, client_weights=None):
    """
    Perform Federated Averaging on client model updates.

    client_updates:
        List of PyTorch state_dict objects.

    client_weights:
        Optional list containing the number of training
        samples for each client.
    """

    if not client_updates:
        raise ValueError("No client updates available for aggregation.")

    # Equal weighting if no weights are provided
    if client_weights is None:
        client_weights = [
            1.0 / len(client_updates)
            for _ in client_updates
        ]

    else:
        total_samples = sum(client_weights)

        if total_samples == 0:
            raise ValueError("Total client weight cannot be zero.")

        client_weights = [
            weight / total_samples
            for weight in client_weights
        ]

    # Start with a copy of the first client's model
    averaged_model = {}

    for parameter_name in client_updates[0]:

        averaged_model[parameter_name] = (
                client_updates[0][parameter_name].clone()
                * client_weights[0]
        )

    # Add remaining clients
    for client_index in range(1, len(client_updates)):

        for parameter_name in averaged_model:

            averaged_model[parameter_name] += (
                    client_updates[client_index][parameter_name]
                    * client_weights[client_index]
            )

    return averaged_model