import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from fl_model import SimpleNN, evaluate_model
from fl_client import FLClient
from aggregation import fed_avg
from secure_aggregation import SecureAggregator


def main():

    print("===================================")
    print("SECURE FL SIMULATION")
    print("===================================")

    # -----------------------------
    # Configuration
    # -----------------------------

    NUM_CLIENTS = 3
    SAMPLES_PER_CLIENT = 1000

    # -----------------------------
    # Load MNIST
    # -----------------------------

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # -----------------------------
    # Create clients
    # -----------------------------

    clients = []

    for i in range(NUM_CLIENTS):

        start = i * SAMPLES_PER_CLIENT
        end = start + SAMPLES_PER_CLIENT

        indices = list(range(start, end))

        client = FLClient(
            client_id=f"Client_{i + 1}",
            dataset=train_dataset,
            indices=indices
        )

        clients.append(client)

    print("\nClients created:")
    for client in clients:
        print(f"- {client.client_id}")

    # -----------------------------
    # Create global model
    # -----------------------------

    global_model = SimpleNN()

    print("\nGlobal model initialized.")

    # -----------------------------
    # Local training
    # -----------------------------

    client_updates = {}

    for client in clients:

        print(
            f"\nTraining {client.client_id}..."
        )

        client.train(global_model)

        update = client.get_model_update()

        client_updates[client.client_id] = update

        print(
            f"{client.client_id} training completed."
        )

    # -----------------------------
    # Secure Aggregation
    # -----------------------------

    client_ids = [
        client.client_id
        for client in clients
    ]

    secure_aggregator = SecureAggregator(
        client_ids
    )

    # Generate masks
    reference_update = next(
        iter(client_updates.values())
    )

    masks = secure_aggregator.create_masks(
        reference_update
    )

    print("\nMasks generated.")

    # Mask updates
    masked_updates = (
        secure_aggregator.create_masked_updates(
            client_updates,
            masks
        )
    )

    print("Client updates masked.")

    # Aggregate masked updates
    secure_sum = (
        secure_aggregator.aggregate_masked_updates(
            masked_updates,
            masks
        )
    )

    print("Secure aggregation completed.")

    # -----------------------------
    # Convert sum into average
    # -----------------------------

    secure_average = {}

    for name in secure_sum:

        secure_average[name] = (
                secure_sum[name] / NUM_CLIENTS
        )

    # -----------------------------
    # Update global model
    # -----------------------------

    global_model.load_state_dict(
        secure_average
    )

    print("Global model updated.")

    # -----------------------------
    # Evaluate
    # -----------------------------

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    accuracy = evaluate_model(
        global_model,
        test_loader
    )

    print("\n===================================")
    print("SECURE FL RESULTS")
    print("===================================")

    print(
        f"Number of clients: {NUM_CLIENTS}"
    )

    print(
        f"Samples per client: {SAMPLES_PER_CLIENT}"
    )

    print(
        f"Test accuracy: {accuracy:.2f}%"
    )

    print("===================================")


if __name__ == "__main__":
    main()