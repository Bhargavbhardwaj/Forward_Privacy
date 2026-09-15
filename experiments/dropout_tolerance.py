import sys
from pathlib import Path
import random

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from torchvision import datasets, transforms

from fl_model import SimpleNN
from fl_client import FLClient
from aggregation import fed_avg


def main():

    print("===================================")
    print("DROPOUT TOLERANCE EXPERIMENT")
    print("===================================")

    number_of_clients = 5
    samples_per_client = 1000

    min_clients = 2

    # --------------------------------
    # Load MNIST
    # --------------------------------

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
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
            dataset=dataset,
            indices=indices
        )

    client_ids = list(
        clients.keys()
    )

    # --------------------------------
    # Test every possible dropout count
    # --------------------------------

    print("\n===================================")
    print("RESULTS")
    print("===================================")

    print(
        f"{'Dropped':<10}"
        f"{'Surviving':<12}"
        f"{'Required':<10}"
        f"{'Status':<12}"
    )

    print("-" * 44)

    for number_of_dropped in range(
            number_of_clients
    ):

        # New global model
        global_model = SimpleNN()

        # Select dropped clients
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

        # --------------------------------
        # Check threshold
        # --------------------------------

        if len(participating_clients) < min_clients:

            print(
                f"{number_of_dropped:<10}"
                f"{len(participating_clients):<12}"
                f"{min_clients:<10}"
                f"{'FAILED':<12}"
            )

            continue

        # --------------------------------
        # Train surviving clients
        # --------------------------------

        client_updates = []
        client_weights = []

        for client_id in participating_clients:

            client = clients[client_id]

            client.train(
                global_model
            )

            update = (
                client.get_model_update()
            )

            client_updates.append(update)

            client_weights.append(
                len(client.dataset)
            )

        # --------------------------------
        # FedAvg
        # --------------------------------

        averaged_model = fed_avg(
            client_updates,
            client_weights
        )

        global_model.load_state_dict(
            averaged_model
        )

        print(
            f"{number_of_dropped:<10}"
            f"{len(participating_clients):<12}"
            f"{min_clients:<10}"
            f"{'SUCCESS':<12}"
        )

    # --------------------------------
    # Final interpretation
    # --------------------------------

    print("\n===================================")
    print("INTERPRETATION")
    print("===================================")

    print(
        f"Total clients: {number_of_clients}"
    )

    print(
        f"Minimum required clients: "
        f"{min_clients}"
    )

    print(
        f"Maximum tolerated dropout: "
        f"{number_of_clients - min_clients} "
        f"clients"
    )

    print(
        f"Maximum tolerated dropout rate: "
        f"{((number_of_clients - min_clients) / number_of_clients) * 100:.0f}%"
    )


if __name__ == "__main__":
    main()