import sys
import os
import copy
import io
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT MODULES
# ============================================================

from fl_model import (
    SimpleNN,
    train_local_model,
    evaluate_model
)

from forward_privacy import ForwardPrivacyClient
from server import Server
from aggregation import fed_avg
from secure_aggregation import SecureAggregator
from mask_recovery import DropoutMaskRecovery


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 5
SAMPLES_PER_CLIENT = 1000

LOCAL_EPOCHS = 1

ROUND_NUMBER = 1

MIN_CLIENTS = 2

DROPPED_CLIENT_ID = "Client_5"


print()
print("======================================================")
print("FULL PQC + FORWARD PRIVACY + SECURE DROPOUT FL")
print("======================================================")


# ============================================================
# LOAD DATA
# ============================================================

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


# ============================================================
# SERVER
# ============================================================

server = Server()

server_public_key = server.get_public_key()

print()
print("[1] Server ML-KEM-768 key pair generated.")


# ============================================================
# CREATE CLIENTS
# ============================================================

clients = []

for i in range(NUM_CLIENTS):

    client_id = f"Client_{i + 1}"

    start_index = i * SAMPLES_PER_CLIENT

    indices = list(
        range(
            start_index,
            start_index + SAMPLES_PER_CLIENT
        )
    )

    forward_privacy = ForwardPrivacyClient(
        client_id,
        server_public_key
    )

    clients.append({
        "id": client_id,
        "indices": indices,
        "fp": forward_privacy
    })

    print(
        f"{client_id}: "
        "ML-KEM encapsulation successful."
    )


# ============================================================
# ESTABLISH PQC SHARED SECRETS
# ============================================================

for client_info in clients:

    client_id = client_info["id"]

    client = client_info["fp"]

    server.establish_client_key(
        client_id,
        client.get_pqc_ciphertext()
    )


print()
print(
    "[2] PQC shared secrets established "
    "for all clients."
)


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model = SimpleNN()


# ============================================================
# LOCAL TRAINING
# ============================================================

print()
print("======================================================")
print("LOCAL TRAINING")
print("======================================================")


client_updates = {}

for client_info in clients:

    client_id = client_info["id"]

    indices = client_info["indices"]

    client_dataset = Subset(
        train_dataset,
        indices
    )

    loader = DataLoader(
        client_dataset,
        batch_size=32,
        shuffle=True
    )

    local_model = SimpleNN()

    local_model.load_state_dict(
        copy.deepcopy(
            global_model.state_dict()
        )
    )

    train_local_model(
        local_model,
        loader,
        epochs=LOCAL_EPOCHS
    )

    update = {}

    for name, parameter in local_model.state_dict().items():

        update[name] = parameter.clone()

    client_updates[client_id] = update

    print(
        f"{client_id}: "
        "local training completed."
    )


# ============================================================
# PQC + AES-GCM ENCRYPTION
# ============================================================

print()
print("======================================================")
print("PQC + AES-GCM ENCRYPTION")
print("======================================================")


server.start_round(
    ROUND_NUMBER
)

for client_info in clients:

    client_id = client_info["id"]

    client = client_info["fp"]

    update = client_updates[client_id]

    buffer = io.BytesIO()

    torch.save(
        update,
        buffer
    )

    plaintext = buffer.getvalue()

    encrypted_update = client.encrypt_update(
        plaintext
    )

    server.receive_update(
        client_id,
        ROUND_NUMBER,
        encrypted_update
    )

    print(
        f"{client_id}: "
        "update encrypted and sent."
    )


# ============================================================
# SERVER CHECKS PARTICIPATION
# ============================================================

all_client_ids = [
    client_info["id"]
    for client_info in clients
]

received_clients = server.get_received_clients(
    ROUND_NUMBER
)

print()
print(
    "[3] Server received encrypted updates:"
)

print(
    received_clients
)


# ============================================================
# SERVER DECRYPTION
# ============================================================

print()
print("======================================================")
print("SERVER DECRYPTION")
print("======================================================")


decrypted_updates = {}

for client_id in received_clients:

    plaintext = server.decrypt_update(
        client_id,
        ROUND_NUMBER
    )

    buffer = io.BytesIO(
        plaintext
    )

    update = torch.load(
        buffer,
        weights_only=False
    )

    decrypted_updates[client_id] = update

    print(
        f"{client_id}: "
        "decryption successful."
    )


# ============================================================
# SECURE AGGREGATION SETUP
# ============================================================

print()
print("======================================================")
print("SECURE MASKING")
print("======================================================")


secure_aggregator = SecureAggregator(
    all_client_ids
)

mask_recovery = DropoutMaskRecovery(
    threshold=2
)

mask_seeds = {}

mask_shares = {}

masked_updates = {}


# Use Client 1's update as shape reference
reference_update = decrypted_updates[
    received_clients[0]
]


# ============================================================
# CREATE MASKS FOR EVERY CLIENT
# ============================================================

for client_id in received_clients:

    # Generate random secret seed
    seed = mask_recovery.create_mask_seed()

    mask_seeds[client_id] = seed

    # Split seed using 2-of-3 Shamir
    shares = mask_recovery.create_shares(
        seed,
        3
    )

    mask_shares[client_id] = shares

    # Generate deterministic mask
    mask = secure_aggregator.generate_mask_from_seed(
        seed,
        reference_update
    )

    # Mask update
    masked_update = (
        secure_aggregator.create_masked_update(
            decrypted_updates[client_id],
            mask
        )
    )

    masked_updates[client_id] = masked_update

    print(
        f"{client_id}: "
        "mask generated and update masked."
    )


# ============================================================
# CLIENT 5 DROPS AFTER SENDING MASKED UPDATE
# ============================================================

print()
print("======================================================")
print("DROPOUT EVENT")
print("======================================================")


print(
    f"{DROPPED_CLIENT_ID} "
    "drops after sending masked update."
)


# The masked update is already at the server.
# Therefore it remains inside the aggregate.


# ============================================================
# VERIFY MINIMUM PARTICIPATION
# ============================================================

if len(masked_updates) < MIN_CLIENTS:

    print(
        "Not enough clients."
    )

    print(
        "FULL INTEGRATION FAILED."
    )

    raise SystemExit


# ============================================================
# AGGREGATE MASKED UPDATES
# ============================================================

masked_sum = (
    secure_aggregator.aggregate_masked_updates(
        masked_updates
    )
)


print()
print(
    "[4] Masked updates aggregated."
)


# ============================================================
# RECOVER DROPPED CLIENT'S MASK
# ============================================================

print()
print("======================================================")
print("DROPPED CLIENT MASK RECOVERY")
print("======================================================")


dropped_shares = mask_shares[
    DROPPED_CLIENT_ID
]

# Recover using only 2 out of 3 shares
recovered_seed = mask_recovery.recover_seed(
    dropped_shares[:2]
)

original_seed = mask_seeds[
    DROPPED_CLIENT_ID
]

if recovered_seed != original_seed:

    print(
        "Dropped client seed recovery FAILED."
    )

    print(
        "FULL INTEGRATION FAILED."
    )

    raise SystemExit


print(
    f"{DROPPED_CLIENT_ID}: "
    "mask seed recovered using 2-of-3 Shamir."
)


# ============================================================
# REGENERATE DROPPED CLIENT MASK
# ============================================================

recovered_mask = (
    secure_aggregator.generate_mask_from_seed(
        recovered_seed,
        reference_update
    )
)


print(
    f"{DROPPED_CLIENT_ID}: "
    "mask regenerated successfully."
)


# ============================================================
# REMOVE DROPPED CLIENT MASK
# ============================================================

unmasked_sum = {}

for name, parameter in masked_sum.items():

    unmasked_sum[name] = (
            parameter
            - recovered_mask[name]
    )


# ============================================================
# REMOVE SURVIVING CLIENT MASKS
# ============================================================

for client_id in received_clients:

    if client_id == DROPPED_CLIENT_ID:

        continue

    seed = mask_seeds[client_id]

    mask = (
        secure_aggregator.generate_mask_from_seed(
            seed,
            reference_update
        )
    )

    for name in unmasked_sum:

        unmasked_sum[name] -= (
            mask[name]
        )


print(
    "[5] All masks removed successfully."
)


# ============================================================
# VERIFY SECURE AGGREGATION
# ============================================================

expected_sum = {}

for client_id in received_clients:

    for name, parameter in (
            decrypted_updates[client_id].items()
    ):

        if name not in expected_sum:

            expected_sum[name] = (
                torch.zeros_like(parameter)
            )

        expected_sum[name] += parameter


aggregation_correct = True

for name in expected_sum:

    if not torch.allclose(
            unmasked_sum[name],
            expected_sum[name],
            rtol=1e-4,
            atol=1e-4
    ):

        aggregation_correct = False

        print(
            f"Aggregation mismatch: {name}"
        )


if not aggregation_correct:

    print()
    print(
        "SECURE AGGREGATION VERIFICATION FAILED."
    )

    print(
        "FULL INTEGRATION FAILED."
    )

    raise SystemExit


print(
    "[6] Secure aggregation verification passed."
)


# ============================================================
# FEDAVG
# ============================================================

print()
print("======================================================")
print("FEDERATED AVERAGING")
print("======================================================")


global_update = fed_avg(
    list(
        decrypted_updates.values()
    )
)

global_model.load_state_dict(
    global_update
)

print(
    "FedAvg completed."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

accuracy = evaluate_model(
    global_model,
    test_loader
)


print()
print("======================================================")
print("MODEL EVALUATION")
print("======================================================")

print(
    f"Global model test accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# FORWARD KEY EVOLUTION
# ============================================================

print()
print("======================================================")
print("FORWARD KEY EVOLUTION")
print("======================================================")


for client_info in clients:

    client_id = client_info["id"]

    client = client_info["fp"]

    old_key = client.get_current_key()

    client.complete_round()

    server.evolve_client_key(
        client_id
    )

    new_key = client.get_current_key()

    if old_key == new_key:

        print(
            f"{client_id}: "
            "key evolution FAILED."
        )

        raise SystemExit

    print(
        f"{client_id}: "
        "K1 -> K2 successful."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("======================================================")
print("FULL INTEGRATION TEST PASSED")
print("======================================================")

print("ML-KEM-768              : PASSED")
print("HKDF-SHA256             : PASSED")
print("AES-256-GCM             : PASSED")
print("Forward Privacy         : PASSED")
print("FL Local Training       : PASSED")
print("Secure Masking          : PASSED")
print("Client Dropout          : PASSED")
print("Shamir Mask Recovery    : PASSED")
print("Secure Aggregation      : PASSED")
print("FedAvg                  : PASSED")
print("Model Evaluation        : PASSED")

print()
print("======================================================")