import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


def train_local_model(model, train_loader, epochs=1):

    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):

        for images, labels in train_loader:

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(
                output,
                labels
            )

            loss.backward()

            optimizer.step()

    return model