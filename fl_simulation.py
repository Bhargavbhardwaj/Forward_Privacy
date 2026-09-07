import torch
from torchvision import datasets, transforms

from fl_model import SimpleNN
from fl_client import FLClient


def main():

    print("Loading MNIST dataset...")

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    # Use only a small subset initially
    total_samples = 6000

    indices = torch.randperm(
        len(dataset)
    )[:total_samples]

    # Divide data among 3 clients
    client_size = total_samples // 3

    clients = []

    for client_id in range(3):

        start = client_id * client_size
        end = start + client_size

        client_indices = indices[
                         start:end
                         ]

        client = FLClient(
            client_id=f"Client_{client_id + 1}",
            dataset=dataset,
            indices=client_indices
        )

        clients.append(client)

        print(
            f"Created Client_{client_id + 1} "
            f"with {len(client_indices)} samples"
        )

    # Global model
    global_model = SimpleNN()

    print("\nStarting local training...")

    for client in clients:

        client.train(
            global_model
        )

        update = client.get_model_update()

        print(
            f"{client.client_id} "
            f"completed local training."
        )

        print(
            f"Number of parameters: "
            f"{len(update)}"
        )


if __name__ == "__main__":
    main()