import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

from fl_model import SimpleNN, evaluate_model
from fl_client import FLClient
from aggregation import fed_avg


def run_round(
        clients,
        global_model,
        participating_clients
):

    client_updates = []
    client_weights = []

    for client_id in participating_clients:

        client = clients[client_id]

        # Train locally
        client.train(global_model)

        # Get trained model parameters
        update = client.get_model_update()

        client_updates.append(update)

        # Number of local samples
        client_weights.append(
            len(client.dataset)
        )

    # FedAvg
    averaged_model = fed_avg(
        client_updates,
        client_weights
    )

    # Update global model
    global_model.load_state_dict(
        averaged_model
    )


def main():

    print("===================================")
    print("DROPOUT VS MODEL ACCURACY")
    print("===================================")

    # -------------------------------
    # Configuration
    # -------------------------------

    number_of_clients = 5
    samples_per_client = 1000

    # -------------------------------
    # Load training data
    # -------------------------------

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # -------------------------------
    # Load test data
    # -------------------------------

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    # -------------------------------
    # Create clients
    # -------------------------------

    clients = {}

    for i in range(number_of_clients):

        start = i * samples_per_client
        end = start + samples_per_client

        indices = list(
            range(start, end)
        )

        client_id = f"Client_{i + 1}"

        clients[client_id] = FLClient(
            client_id=client_id,
            dataset=train_dataset,
            indices=indices
        )

    client_ids = list(
        clients.keys()
    )

    # -------------------------------
    # Test different dropout rates
    # -------------------------------

    dropout_rates = [
        0.0,
        0.2,
        0.4,
        0.6
    ]

    print("\n===================================")
    print("RESULTS")
    print("===================================")

    print(
        f"{'Dropout':<12}"
        f"{'Clients':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 36)

    for dropout_rate in dropout_rates:

        # New global model
        global_model = SimpleNN()

        # Determine number of dropped clients
        number_of_dropped = int(
            number_of_clients *
            dropout_rate
        )

        # Select dropped clients
        import random

        dropped_clients = set(
            random.sample(
                client_ids,
                number_of_dropped
            )
        )

        participating_clients = [
            client_id
            for client_id in client_ids
            if client_id
               not in dropped_clients
        ]

        # Run one FL round
        run_round(
            clients,
            global_model,
            participating_clients
        )

        # Evaluate global model
        accuracy = evaluate_model(
            global_model,
            test_loader
        )

        print(
            f"{dropout_rate * 100:<12.0f}"
            f"{len(participating_clients):<12}"
            f"{accuracy:<12.2f}"
        )


if __name__ == "__main__":
    main()