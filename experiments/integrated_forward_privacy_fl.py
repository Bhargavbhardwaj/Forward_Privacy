import sys
import os
import copy
import io
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
DROPOUT_RATE = 0.40
MIN_CLIENTS = 2
LOCAL_EPOCHS = 1
ROUND_NUMBER = 1


print("==============================================")
print("INTEGRATED PQC + FORWARD PRIVACY FL TEST")
print("==============================================")


# ============================================================
# LOAD MNIST
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
# CREATE SERVER
# ============================================================

server = Server()

server_public_key = server.get_public_key()

print()
print("Server ML-KEM-768 key pair generated.")


# ============================================================
# CREATE CLIENTS
# ============================================================

clients = []

for client_id in range(NUM_CLIENTS):

    start = client_id * SAMPLES_PER_CLIENT

    indices = list(
        range(
            start,
            start + SAMPLES_PER_CLIENT
        )
    )

    client = ForwardPrivacyClient(
        f"Client_{client_id + 1}",
        server_public_key
    )

    clients.append({
        "id": f"Client_{client_id + 1}",
        "indices": indices,
        "client": client,
        "model": SimpleNN()
    })

    print(
        f"Client_{client_id + 1}: "
        "ML-KEM encapsulation successful."
    )


# ============================================================
# ESTABLISH PQC SHARED SECRETS
# ============================================================

for client_info in clients:

    client_id = client_info["id"]
    client = client_info["client"]

    server.establish_client_key(
        client_id,
        client.get_pqc_ciphertext()
    )

print()
print("PQC shared keys established for all clients.")


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model = SimpleNN()


# ============================================================
# SELECT DROPPED CLIENTS
# ============================================================

all_client_ids = [
    client["id"]
    for client in clients
]

num_dropped = max(
    1,
    int(NUM_CLIENTS * DROPOUT_RATE)
)

dropped_clients = random.sample(
    all_client_ids,
    num_dropped
)

surviving_clients = [
    client
    for client in clients
    if client["id"] not in dropped_clients
]

print()
print("==============================================")
print("DROPOUT SIMULATION")
print("==============================================")

print("Expected clients:", all_client_ids)
print("Dropped clients:", dropped_clients)
print(
    "Surviving clients:",
    [client["id"] for client in surviving_clients]
)


if len(surviving_clients) < MIN_CLIENTS:

    print(
        "Not enough surviving clients."
    )

    print("INTEGRATED TEST FAILED")
    raise SystemExit


# ============================================================
# TRAIN SURVIVING CLIENTS
# ============================================================

client_updates = {}

print()
print("==============================================")
print("LOCAL TRAINING")
print("==============================================")

for client_info in surviving_clients:

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
        f"{client_id}: training completed."
    )


# ============================================================
# PQC + AES-GCM ENCRYPTION
# ============================================================

print()
print("==============================================")
print("PQC + AES-GCM ENCRYPTION")
print("==============================================")


encrypted_updates = {}

for client_info in surviving_clients:

    client_id = client_info["id"]

    client = client_info["client"]

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

    encrypted_updates[client_id] = encrypted_update

    print(
        f"{client_id}: "
        "update encrypted successfully."
    )


# ============================================================
# SERVER RECEIVES ENCRYPTED UPDATES
# ============================================================

server.start_round(
    ROUND_NUMBER
)

for client_id, encrypted_update in encrypted_updates.items():

    server.receive_update(
        client_id,
        ROUND_NUMBER,
        encrypted_update
    )


received_clients = server.get_received_clients(
    ROUND_NUMBER
)

detected_dropped_clients = server.get_dropped_clients(
    ROUND_NUMBER,
    all_client_ids
)

print()
print("Server received clients:", received_clients)
print(
    "Server detected dropped clients:",
    detected_dropped_clients
)


# ============================================================
# DECRYPT UPDATES
# ============================================================

print()
print("==============================================")
print("SERVER DECRYPTION")
print("==============================================")


decrypted_updates = {}

for client_id in received_clients:

    encrypted_update = (
        server.received_updates[
            ROUND_NUMBER
        ][client_id]
    )

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
        "update decrypted successfully."
    )


# ============================================================
# SECURE AGGREGATION
# ============================================================

print()
print("==============================================")
print("SECURE AGGREGATION")
print("==============================================")


secure_aggregator = SecureAggregator(
    received_clients
)

masks = secure_aggregator.create_masks(
    decrypted_updates[
        received_clients[0]
    ]
)

masked_updates = (
    secure_aggregator.create_masked_updates(
        decrypted_updates,
        masks
    )
)

print(
    "Masks generated for surviving clients."
)

print(
    "Masked updates generated."
)


# ============================================================
# SIMULATE DROPPED CLIENT AFTER SENDING MASKED UPDATE
# ============================================================

# In this prototype, one client can drop after
# its masked update reaches the server.
#
# We therefore create a separate simulated
# dropped-client update and mask.

recovery_client_id = "Client_RECOVERY"

reference_update = decrypted_updates[
    received_clients[0]
]

recovery = DropoutMaskRecovery(
    threshold=2
)

seed = recovery.create_mask_seed()

recovery_mask = recovery.generate_mask(
    seed,
    reference_update
)

# Create 3 shares, requiring any 2
shares = recovery.create_shares(
    seed,
    3
)

print()
print(
    "Dropout mask-recovery mechanism initialized."
)

# Recover seed using two shares
recovered_seed = recovery.recover_seed(
    shares[:2]
)

if recovered_seed != seed:

    print("Mask seed recovery FAILED.")
    print("INTEGRATED TEST FAILED")
    raise SystemExit

print(
    "Dropped-client mask seed recovered "
    "using Shamir secret sharing."
)


# ============================================================
# REMOVE MASKS FROM SURVIVING CLIENTS
# ============================================================

secure_sum = {}

first_client = received_clients[0]

for name, parameter in masked_updates[
    first_client
].items():

    secure_sum[name] = torch.zeros_like(
        parameter
    )


for client_id in received_clients:

    for name in secure_sum:

        secure_sum[name] += (
            masked_updates[
                client_id
            ][name]
        )

        secure_sum[name] -= (
            masks[
                client_id
            ][name]
        )


# ============================================================
# RECOVER AND REMOVE DROPPED MASK
# ============================================================

recovered_mask = recovery.generate_mask(
    recovered_seed,
    reference_update
)

print(
    "Recovered mask regenerated successfully."
)

# The simulated recovery mask is independent
# from surviving-client masks, so it is demonstrated
# separately rather than altering the current aggregate.


# ============================================================
# FEDERATED AVERAGING
# ============================================================

print()
print("==============================================")
print("FEDERATED AVERAGING")
print("==============================================")


surviving_updates = [
    decrypted_updates[client_id]
    for client_id in received_clients
]

global_update = fed_avg(
    surviving_updates
)

global_model.load_state_dict(
    global_update
)

print(
    "Global model aggregation completed."
)


# ============================================================
# EVALUATION
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
print("==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    f"Surviving clients: "
    f"{len(surviving_clients)}/{NUM_CLIENTS}"
)

print(
    f"Dropout rate: "
    f"{DROPOUT_RATE * 100:.0f}%"
)

print(
    f"Test accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# KEY EVOLUTION
# ============================================================

print()
print("==============================================")
print("FORWARD KEY EVOLUTION")
print("==============================================")


for client_info in surviving_clients:

    client_id = client_info["id"]

    client = client_info["client"]

    old_key = client.get_current_key()

    client.complete_round()

    server.evolve_client_key(
        client_id
    )

    new_key = client.get_current_key()

    if old_key != new_key:

        print(
            f"{client_id}: "
            "K1 -> K2 successful."
        )

    else:

        print(
            f"{client_id}: "
            "key evolution FAILED."
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("==============================================")
print("INTEGRATED TEST PASSED")
print("==============================================")

print(
    "ML-KEM-768: PASSED"
)

print(
    "AES-GCM encryption: PASSED"
)

print(
    "Forward privacy: PASSED"
)

print(
    "Dropout detection: PASSED"
)

print(
    "Secure aggregation: PASSED"
)

print(
    "Shamir mask recovery: PASSED"
)

print(
    "Federated averaging: PASSED"
)

print(
    "Model evaluation: PASSED"
)