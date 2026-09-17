import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import torch
from torchvision import datasets, transforms

from fl_model import SimpleNN, evaluate_model
from fl_client import FLClient
from secure_aggregation import SecureAggregator


def main():

    print("===================================")
    print("SECURE FL + DROPOUT SIMULATION")
    print("===================================")

    # -----------------------------
    # Configuration
    # -----------------------------

    NUM_CLIENTS = 5
    SAMPLES_PER_CLIENT = 1000
    DROPOUT_RATE = 0.40
    MIN_CLIENTS = 2

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
    # Simulate dropout
    # -----------------------------

    num_dropped = int(
        NUM_CLIENTS * DROPOUT_RATE
    )

    dropped_clients = random.sample(
        clients,
        num_dropped
    )

    dropped_ids = {
        client.client_id
        for client in dropped_clients
    }

    surviving_clients = [
        client
        for client in clients
        if client.client_id not in dropped_ids
    ]

    print("\nDropped clients:")

    for client in dropped_clients:
        print(f"- {client.client_id}")

    print("\nSurviving clients:")

    for client in surviving_clients:
        print(f"- {client.client_id}")

    # -----------------------------
    # Check minimum participation
    # -----------------------------

    if len(surviving_clients) < MIN_CLIENTS:

        print("\nNot enough clients.")
        print("Secure aggregation cannot proceed.")

        return

    # -----------------------------
    # Global model
    # -----------------------------

    global_model = SimpleNN()

    print("\nGlobal model initialized.")

    # -----------------------------
    # Local training
    # -----------------------------

    client_updates = {}

    for client in surviving_clients:

        print(
            f"\nTraining {client.client_id}..."
        )

        client.train(global_model)

        client_updates[
            client.client_id
        ] = client.get_model_update()

        print(
            f"{client.client_id} training completed."
        )

    # -----------------------------
    # Secure Aggregation
    # -----------------------------

    surviving_ids = [
        client.client_id
        for client in surviving_clients
    ]

    secure_aggregator = SecureAggregator(
        surviving_ids
    )

    reference_update = next(
        iter(client_updates.values())
    )

    masks = secure_aggregator.create_masks(
        reference_update
    )

    print("\nMasks generated for surviving clients.")

    masked_updates = (
        secure_aggregator.create_masked_updates(
            client_updates,
            masks
        )
    )

    print("Client updates masked.")

    secure_sum = (
        secure_aggregator.aggregate_masked_updates(
            masked_updates,
            masks
        )
    )

    print("Secure aggregation completed.")

    # -----------------------------
    # Average surviving updates
    # -----------------------------

    secure_average = {}

    for name in secure_sum:

        secure_average[name] = (
                secure_sum[name]
                / len(surviving_clients)
        )

    # -----------------------------
    # Update global model
    # -----------------------------

    global_model.load_state_dict(
        secure_average
    )

    print("Global model updated.")

    # -----------------------------
    # Evaluation
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

    # -----------------------------
    # Results
    # -----------------------------

    print("\n===================================")
    print("SECURE FL + DROPOUT RESULTS")
    print("===================================")

    print(
        f"Total clients: {NUM_CLIENTS}"
    )

    print(
        f"Dropped clients: {len(dropped_clients)}"
    )

    print(
        f"Surviving clients: {len(surviving_clients)}"
    )

    print(
        f"Dropout rate: {DROPOUT_RATE * 100:.0f}%"
    )

    print(
        f"Test accuracy: {accuracy:.2f}%"
    )

    print("===================================")


if __name__ == "__main__":
    main()