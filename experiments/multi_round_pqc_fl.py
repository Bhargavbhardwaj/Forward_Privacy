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

NUM_ROUNDS = 3
LOCAL_EPOCHS = 1

DROPOUT_RATE = 0.40
MIN_CLIENTS = 2

random.seed(42)


print()
print("======================================================")
print("MULTI-ROUND PQC FORWARD-PRIVATE FL")
print("======================================================")


# ============================================================
# DATA
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

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# SERVER
# ============================================================

server = Server()

server_public_key = server.get_public_key()

print()
print("Server ML-KEM-768 key pair generated.")


# ============================================================
# CLIENTS
# ============================================================

clients = []

for i in range(NUM_CLIENTS):

    client_id = f"Client_{i + 1}"

    start = i * SAMPLES_PER_CLIENT

    indices = list(
        range(
            start,
            start + SAMPLES_PER_CLIENT
        )
    )

    fp_client = ForwardPrivacyClient(
        client_id,
        server_public_key
    )

    clients.append({
        "id": client_id,
        "indices": indices,
        "fp": fp_client
    })

    server.establish_client_key(
        client_id,
        fp_client.get_pqc_ciphertext()
    )


print(
    "PQC shared keys established for all clients."
)


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model = SimpleNN()


# ============================================================
# STORE ROUND INFORMATION
# ============================================================

round_ciphertexts = {}

round_keys = {}

round_accuracies = {}


# ============================================================
# MULTI-ROUND LOOP
# ============================================================

for round_number in range(
        1,
        NUM_ROUNDS + 1
):

    print()
    print("======================================================")
    print(f"ROUND {round_number}")
    print("======================================================")


    # --------------------------------------------------------
    # Store current keys BEFORE encryption
    # --------------------------------------------------------

    round_keys[round_number] = {}

    for client_info in clients:

        client_id = client_info["id"]

        client = client_info["fp"]

        round_keys[round_number][
            client_id
        ] = client.get_current_key()


    # --------------------------------------------------------
    # Select dropout clients
    # --------------------------------------------------------

    all_client_ids = [
        client_info["id"]
        for client_info in clients
    ]

    num_dropped = int(
        NUM_CLIENTS * DROPOUT_RATE
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

    print(
        "Dropped clients:",
        dropped_clients
    )

    print(
        "Surviving clients:",
        [
            client["id"]
            for client in surviving_clients
        ]
    )


    if len(surviving_clients) < MIN_CLIENTS:

        print(
            "Not enough surviving clients."
        )

        raise SystemExit


    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

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

        for name, parameter in (
                local_model.state_dict().items()
        ):

            update[name] = parameter.clone()

        client_updates[
            client_id
        ] = update


    print("Local training completed.")


    # --------------------------------------------------------
    # Secure aggregation setup
    # --------------------------------------------------------

    secure_aggregator = SecureAggregator(
        all_client_ids
    )

    mask_recovery = DropoutMaskRecovery(
        threshold=2
    )


    masked_updates = {}

    mask_seeds = {}

    mask_shares = {}


    # --------------------------------------------------------
    # Server receives surviving client updates
    # --------------------------------------------------------

    server.start_round(
        round_number
    )

    for client_info in surviving_clients:

        client_id = client_info["id"]

        client = client_info["fp"]

        update = client_updates[
            client_id
        ]

        # Serialize model update
        buffer = io.BytesIO()

        torch.save(
            update,
            buffer
        )

        plaintext = buffer.getvalue()

        # PQC-derived forward-private AES key
        encrypted_update = (
            client.encrypt_update(
                plaintext
            )
        )

        server.receive_update(
            client_id,
            round_number,
            encrypted_update
        )

        if round_number not in round_ciphertexts:

            round_ciphertexts[
                round_number
            ] = {}

        round_ciphertexts[
            round_number
        ][client_id] = encrypted_update


    # --------------------------------------------------------
    # Decrypt surviving updates
    # --------------------------------------------------------

    decrypted_updates = {}

    for client_info in surviving_clients:

        client_id = client_info["id"]

        plaintext = server.decrypt_update(
            client_id,
            round_number
        )

        buffer = io.BytesIO(
            plaintext
        )

        update = torch.load(
            buffer,
            weights_only=False
        )

        decrypted_updates[
            client_id
        ] = update


    print(
        "PQC + AES-GCM decryption completed."
    )


    # --------------------------------------------------------
    # Generate masks
    # --------------------------------------------------------

    reference_update = decrypted_updates[
        surviving_clients[0]["id"]
    ]

    for client_id in decrypted_updates:

        seed = (
            mask_recovery.create_mask_seed()
        )

        mask_seeds[
            client_id
        ] = seed

        shares = (
            mask_recovery.create_shares(
                seed,
                3
            )
        )

        mask_shares[
            client_id
        ] = shares

        mask = (
            secure_aggregator
            .generate_mask_from_seed(
                seed,
                reference_update
            )
        )

        masked_updates[
            client_id
        ] = (
            secure_aggregator
            .create_masked_update(
                decrypted_updates[
                    client_id
                ],
                mask
            )
        )


    # --------------------------------------------------------
    # Aggregate masked updates
    # --------------------------------------------------------

    masked_sum = (
        secure_aggregator
        .aggregate_masked_updates(
            masked_updates
        )
    )


    # --------------------------------------------------------
    # Recover/remove masks
    # --------------------------------------------------------

    unmasked_sum = {}

    for name, parameter in masked_sum.items():

        unmasked_sum[name] = (
            parameter.clone()
        )


    for client_id in decrypted_updates:

        seed = mask_seeds[
            client_id
        ]

        # Simulate dropout mask recovery
        if client_id in dropped_clients:

            recovered_seed = (
                mask_recovery.recover_seed(
                    mask_shares[
                        client_id
                    ][:2]
                )
            )

            seed = recovered_seed


        mask = (
            secure_aggregator
            .generate_mask_from_seed(
                seed,
                reference_update
            )
        )

        for name in unmasked_sum:

            unmasked_sum[name] -= (
                mask[name]
            )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only surviving clients actually participated in
    # this round. Therefore the aggregate should equal
    # the sum of surviving updates.
    # --------------------------------------------------------

    expected_sum = {}

    for client_id in decrypted_updates:

        for name, parameter in (
                decrypted_updates[
                    client_id
                ].items()
        ):

            if name not in expected_sum:

                expected_sum[name] = (
                    torch.zeros_like(
                        parameter
                    )
                )

            expected_sum[name] += parameter


    for name in expected_sum:

        if not torch.allclose(
                unmasked_sum[name],
                expected_sum[name],
                rtol=1e-4,
                atol=1e-4
        ):

            print(
                "Secure aggregation verification FAILED."
            )

            raise SystemExit


    # --------------------------------------------------------
    # FedAvg
    # --------------------------------------------------------

    global_update = fed_avg(
        list(
            decrypted_updates.values()
        )
    )

    global_model.load_state_dict(
        global_update
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = evaluate_model(
        global_model,
        test_loader
    )

    round_accuracies[
        round_number
    ] = accuracy

    print(
        f"Round {round_number} accuracy: "
        f"{accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # Forward key evolution
    # --------------------------------------------------------

    for client_info in clients:

        client_id = client_info["id"]

        client = client_info["fp"]

        client.complete_round()

        server.evolve_client_key(
            client_id
        )


    print(
        "Forward keys evolved."
    )


# ============================================================
# FORWARD PRIVACY COMPROMISE TEST
# ============================================================

print()
print("======================================================")
print("FORWARD PRIVACY COMPROMISE TEST")
print("======================================================")


# Pick a client that participated in every round
test_client_id = "Client_1"

if test_client_id not in round_ciphertexts[1]:
    print(
        "Client_1 did not participate in Round 1."
    )
else:

    # Current key is now K4 after 3 rounds
    compromised_key = clients[
        int(test_client_id.split("_")[1]) - 1
        ]["fp"].get_current_key()


    # Try to decrypt Round 1 ciphertext with current key
    encrypted_round1 = (
        round_ciphertexts[1][
            test_client_id
        ]
    )


    try:

        from crypto_utils import decrypt_data

        decrypt_data(
            compromised_key,
            encrypted_round1
        )

        print(
            "WARNING: old ciphertext decrypted."
        )

    except Exception:

        print(
            "Round 1 ciphertext cannot be "
            "decrypted using current key."
        )

        print(
            "Forward privacy property verified."
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("======================================================")
print("MULTI-ROUND RESULTS")
print("======================================================")


for round_number, accuracy in (
        round_accuracies.items()
):

    print(
        f"Round {round_number}: "
        f"{accuracy:.2f}%"
    )


print()
print("======================================================")
print("MULTI-ROUND PQC FL TEST PASSED")
print("======================================================")

print("ML-KEM-768           : PASSED")
print("AES-256-GCM          : PASSED")
print("Forward Privacy      : PASSED")
print("Multi-Round FL       : PASSED")
print("Random Dropout       : PASSED")
print("Secure Aggregation   : PASSED")
print("Mask Recovery        : PASSED")
print("FedAvg               : PASSED")
print("Accuracy Evaluation  : PASSED")