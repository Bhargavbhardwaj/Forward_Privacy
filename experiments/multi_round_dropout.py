import sys
from pathlib import Path
import random
import time

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from fl_model import SimpleNN, evaluate_model
from fl_client import FLClient
from aggregation import fed_avg


def main():

    print("===================================")
    print("MULTI-ROUND DROPOUT-RESILIENT FL")
    print("===================================")

    # --------------------------------
    # Configuration
    # --------------------------------

    number_of_clients = 5
    samples_per_client = 1000

    number_of_rounds = 10

    dropout_rate = 0.40

    min_clients = 2

    print(f"\nClients: {number_of_clients}")
    print(f"Rounds: {number_of_rounds}")
    print(
        f"Dropout rate: "
        f"{dropout_rate * 100:.0f}%"
    )
    print(
        f"Minimum clients required: "
        f"{min_clients}"
    )

    # --------------------------------
    # Load MNIST
    # --------------------------------

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    # --------------------------------
    # Create clients
    # --------------------------------

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

    print("\nClients created successfully.")

    # --------------------------------
    # Global model
    # --------------------------------

    global_model = SimpleNN()

    # --------------------------------
    # Initial accuracy
    # --------------------------------

    initial_accuracy = evaluate_model(
        global_model,
        test_loader
    )

    print(
        f"\nInitial accuracy: "
        f"{initial_accuracy:.2f}%"
    )

    # --------------------------------
    # FL rounds
    # --------------------------------

    results = []

    for round_number in range(
            1,
            number_of_rounds + 1
    ):

        print("\n===================================")
        print(
            f"ROUND {round_number}"
        )
        print("===================================")

        # ----------------------------
        # Select dropped clients
        # ----------------------------

        number_of_dropped = int(
            number_of_clients *
            dropout_rate
        )

        dropped_clients = set(
            random.sample(
                client_ids,
                number_of_dropped
            )
        )

        participating_clients = [
            client_id
            for client_id in client_ids
            if client_id not in dropped_clients
        ]

        print(
            f"Participating clients: "
            f"{participating_clients}"
        )

        print(
            f"Dropped clients: "
            f"{list(dropped_clients)}"
        )

        # ----------------------------
        # Check threshold
        # ----------------------------

        if len(participating_clients) < min_clients:

            print(
                "ROUND FAILED: "
                "Not enough clients."
            )

            results.append({
                "round": round_number,
                "participating": len(
                    participating_clients
                ),
                "dropped": len(
                    dropped_clients
                ),
                "accuracy": None,
                "status": "FAILED"
            })

            continue

        # ----------------------------
        # Local training
        # ----------------------------

        start_time = time.perf_counter()

        client_updates = []
        client_weights = []

        for client_id in participating_clients:

            client = clients[client_id]

            print(
                f"{client_id}: "
                f"local training..."
            )

            client.train(
                global_model
            )

            update = (
                client.get_model_update()
            )

            client_updates.append(
                update
            )

            client_weights.append(
                len(client.dataset)
            )

        # ----------------------------
        # FedAvg
        # ----------------------------

        averaged_model = fed_avg(
            client_updates,
            client_weights
        )

        global_model.load_state_dict(
            averaged_model
        )

        end_time = time.perf_counter()

        round_time = (
                end_time - start_time
        )

        # ----------------------------
        # Evaluate global model
        # ----------------------------

        accuracy = evaluate_model(
            global_model,
            test_loader
        )

        print(
            f"\nRound {round_number} "
            f"completed."
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Round time: "
            f"{round_time:.2f} seconds"
        )

        results.append({
            "round": round_number,
            "participating": len(
                participating_clients
            ),
            "dropped": len(
                dropped_clients
            ),
            "accuracy": accuracy,
            "status": "SUCCESS"
        })

    # --------------------------------
    # Final summary
    # --------------------------------

    print("\n\n===================================")
    print("MULTI-ROUND EXPERIMENT SUMMARY")
    print("===================================")

    print(
        f"{'Round':<8}"
        f"{'Clients':<10}"
        f"{'Dropped':<10}"
        f"{'Accuracy':<12}"
        f"{'Status':<10}"
    )

    print("-" * 50)

    for result in results:

        accuracy = result["accuracy"]

        if accuracy is None:
            accuracy_text = "N/A"
        else:
            accuracy_text = (
                f"{accuracy:.2f}%"
            )

        print(
            f"{result['round']:<8}"
            f"{result['participating']:<10}"
            f"{result['dropped']:<10}"
            f"{accuracy_text:<12}"
            f"{result['status']:<10}"
        )


if __name__ == "__main__":
    main()