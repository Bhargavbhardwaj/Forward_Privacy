import os
import sys
import copy
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ==========================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================
# PROJECT IMPORTS
# ==========================================

from fl_model import (
    SimpleNN,
    train_local_model,
    evaluate_model
)

from mask_recovery import DropoutMaskRecovery


# ==========================================
# CONFIGURATION
# ==========================================

NUM_CLIENTS = 5

SAMPLES_PER_CLIENT = 1000

BATCH_SIZE = 32

EPOCHS = 1

DROPPED_CLIENT = "Client_5"

RANDOM_SEED = 42


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


# ==========================================
# LOAD MNIST
# ==========================================

print("===================================")
print("SECURE DROPOUT FL MASK RECOVERY")
print("===================================")
print()

print("Loading MNIST dataset...")


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


print(
    f"MNIST training samples: "
    f"{len(train_dataset)}"
)

print()


# ==========================================
# CREATE CLIENT DATA
# ==========================================

client_datasets = {}

available_indices = list(
    range(len(train_dataset))
)

random.shuffle(available_indices)


for client_number in range(NUM_CLIENTS):

    start = (
            client_number
            * SAMPLES_PER_CLIENT
    )

    end = (
            start
            + SAMPLES_PER_CLIENT
    )

    indices = available_indices[
              start:end
              ]

    client_datasets[
        f"Client_{client_number + 1}"
    ] = Subset(
        train_dataset,
        indices
    )


# ==========================================
# INITIAL GLOBAL MODEL
# ==========================================

global_model = SimpleNN()


# ==========================================
# TRAIN CLIENT MODELS
# ==========================================

client_models = {}

print("Starting local training...")
print()


for client_id in client_datasets:

    print(
        f"{client_id}: "
        "training local model..."
    )

    model = SimpleNN()

    model.load_state_dict(
        copy.deepcopy(
            global_model.state_dict()
        )
    )

    loader = DataLoader(
        client_datasets[client_id],
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_local_model(
        model,
        loader,
        epochs=EPOCHS
    )

    client_models[client_id] = (
        copy.deepcopy(
            model.state_dict()
        )
    )

    print(
        f"{client_id}: "
        "training complete."
    )


print()
print("All client updates created.")
print()


# ==========================================
# CREATE MASK RECOVERY SYSTEM
# ==========================================

mask_recovery = DropoutMaskRecovery(
    threshold=2
)


# ==========================================
# GENERATE MASK SEEDS
# ==========================================

mask_seeds = {}

for client_id in client_models:

    mask_seeds[client_id] = (
        mask_recovery.create_mask_seed()
    )


print("Mask seeds generated.")


# ==========================================
# GENERATE MASKS
# ==========================================

reference_update = (
    client_models["Client_1"]
)

masks = {}

for client_id in client_models:

    masks[client_id] = (
        mask_recovery.generate_mask(
            mask_seeds[client_id],
            reference_update
        )
    )


print("Masks generated.")


# ==========================================
# CREATE SECRET SHARING RECOVERY SHARES
# ==========================================

recovery_shares = {}

for client_id in client_models:

    recovery_shares[client_id] = (
        mask_recovery.create_shares(
            mask_seeds[client_id],
            num_shares=3
        )
    )


print("Recovery shares created.")
print()


# ==========================================
# MASK CLIENT UPDATES
# ==========================================

masked_updates = {}

for client_id in client_models:

    masked_updates[client_id] = {}

    for name in client_models[client_id]:

        masked_updates[client_id][name] = (
                client_models[client_id][name]
                + masks[client_id][name]
        )


print("All client updates masked.")
print()


# ==========================================
# SIMULATE DROPOUT
# ==========================================

print(
    f"{DROPPED_CLIENT} "
    "drops after sending its masked update."
)

print()


surviving_clients = [
    client_id
    for client_id in client_models
    if client_id != DROPPED_CLIENT
]


# ==========================================
# CREATE MASKED AGGREGATE
# ==========================================

masked_aggregate = {}

for name in reference_update:

    masked_aggregate[name] = (
        torch.zeros_like(
            reference_update[name]
        )
    )


# IMPORTANT:
#
# The dropped client already sent
# its masked update.
#
# Therefore its update remains
# in the aggregate.

for client_id in client_models:

    for name in masked_aggregate:

        masked_aggregate[name] += (
            masked_updates[client_id][name]
        )


print("Masked aggregate created.")


# ==========================================
# REMOVE SURVIVING CLIENT MASKS
# ==========================================

for client_id in surviving_clients:

    for name in masked_aggregate:

        masked_aggregate[name] -= (
            masks[client_id][name]
        )


print(
    "Surviving client masks removed."
)


# ==========================================
# RECOVER DROPPED CLIENT SEED
# ==========================================

dropped_shares = (
    recovery_shares[DROPPED_CLIENT]
)


recovery_shares_used = [
    dropped_shares[0],
    dropped_shares[1]
]


recovered_seed = (
    mask_recovery.recover_seed(
        recovery_shares_used
    )
)


print(
    "Dropped client mask seed recovered."
)


# ==========================================
# REGENERATE DROPPED CLIENT MASK
# ==========================================

recovered_mask = (
    mask_recovery.generate_mask(
        recovered_seed,
        reference_update
    )
)


print(
    "Dropped client mask regenerated."
)


# ==========================================
# REMOVE DROPPED CLIENT MASK
# ==========================================

for name in masked_aggregate:

    masked_aggregate[name] -= (
        recovered_mask[name]
    )


print(
    "Dropped client mask removed."
)

print()


# ==========================================
# AVERAGE AGGREGATE
# ==========================================

global_model.load_state_dict(
    copy.deepcopy(
        global_model.state_dict()
    )
)


new_global_state = {}

num_received_clients = len(
    client_models
)


for name in masked_aggregate:

    new_global_state[name] = (
            masked_aggregate[name]
            / num_received_clients
    )


global_model.load_state_dict(
    new_global_state
)


# ==========================================
# EVALUATE GLOBAL MODEL
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


accuracy = evaluate_model(
    global_model,
    test_loader
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("===================================")
print("EXPERIMENT RESULTS")
print("===================================")

print(
    f"Total clients: {NUM_CLIENTS}"
)

print(
    f"Dropped client: {DROPPED_CLIENT}"
)

print(
    f"Received updates: "
    f"{num_received_clients}"
)

print(
    f"Surviving clients: "
    f"{len(surviving_clients)}"
)

print(
    f"Global model accuracy: "
    f"{accuracy:.2f}%"
)

print()

print("===================================")
print(
    "SECURE DROPOUT MASK RECOVERY "
    "EXPERIMENT PASSED"
)
print("===================================")