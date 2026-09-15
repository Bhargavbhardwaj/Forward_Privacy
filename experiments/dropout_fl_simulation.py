import sys
from pathlib import Path
import random
import time

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from torchvision import datasets, transforms

from fl_model import SimpleNN
from fl_client import FLClient
from aggregation import fed_avg


def run_experiment(
        clients,
        global_model,
        client_ids,
        dropout_rate,
        min_clients
):

    number_of_clients = len(client_ids)

    number_of_dropped = int(
        number_of_clients * dropout_rate
    )

    dropped_clients = set(
        random.sample(
            client_ids,
            number_of_dropped
        )
    )

    surviving_clients = [
        client_id
        for client_id in client_ids
        if client_id not in dropped_clients
    ]

    # Check minimum participation
    if len(surviving_clients) < min_clients:

        return {
            "dropout_rate": dropout_rate,
            "dropped": len(dropped_clients),
            "surviving": len(surviving_clients),
            "status": "FAILED",
            "training_time": 0
        }

    # Local training
    client_updates = []
    client_weights = []

    start_time = time.perf_counter()

    for client_id in surviving_clients:

        client = clients[client_id]

        client.train(global_model)

        update = client.get_model_update()

        client_updates.append(update)

        client_weights.append(
            len(client.dataset)
        )

    # FedAvg
    averaged_update = fed_avg(
        client_updates,
        client_weights
    )

    # Update global model
    global_model.load_state_dict(
        averaged_update
    )

    end_time = time.perf_counter()

    return {
        "dropout_rate": dropout_rate,
        "dropped": len(dropped_clients),
        "surviving": len(surviving_clients),
        "status": "SUCCESS",
        "training_time": end_time - start_time
    }


def main():

    print("===================================")
    print("DROPOUT RATE EXPERIMENT")
    print("===================================")

    # -------------------------------
    # Configuration
    # -------------------------------

    number_of_clients = 5
    samples_per_client = 1000
    min_clients = 2

    dropout_rates = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8
    ]

    # -------------------------------
    # Load MNIST
    # -------------------------------

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # -------------------------------
    # Create clients
    # -------------------------------

    clients = {}

    for i in range(number_of_clients):

        start = i * samples_per_client
        end = start + samples_per_client

        indices = list(range(start, end))

        client_id = f"Client_{i + 1}"

        clients[client_id] = FLClient(
            client_id=client_id,
            dataset=dataset,
            indices=indices
        )

    client_ids = list(clients.keys())

    # -------------------------------
    # Run experiments
    # -------------------------------

    results = []

    for dropout_rate in dropout_rates:

        print(
            f"\nTesting dropout rate: "
            f"{dropout_rate * 100:.0f}%"
        )

        # New global model for each experiment
        global_model = SimpleNN()

        result = run_experiment(
            clients,
            global_model,
            client_ids,
            dropout_rate,
            min_clients
        )

        results.append(result)

        print(
            f"Surviving clients: "
            f"{result['surviving']}"
        )

        print(
            f"Dropped clients: "
            f"{result['dropped']}"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        if result["status"] == "SUCCESS":

            print(
                f"Training time: "
                f"{result['training_time']:.2f} seconds"
            )

    # -------------------------------
    # Summary
    # -------------------------------

    print("\n===================================")
    print("EXPERIMENT SUMMARY")
    print("===================================")

    print(
        f"{'Dropout':<12}"
        f"{'Surviving':<12}"
        f"{'Dropped':<10}"
        f"{'Status':<10}"
        f"{'Time (s)':<10}"
    )

    print("-" * 54)

    for result in results:

        print(
            f"{result['dropout_rate'] * 100:<12.0f}"
            f"{result['surviving']:<12}"
            f"{result['dropped']:<10}"
            f"{result['status']:<10}"
            f"{result['training_time']:<10.2f}"
        )


if __name__ == "__main__":
    main()