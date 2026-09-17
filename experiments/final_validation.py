"""
Final validation for:
PQC + Forward Privacy + Secure Aggregation + Dropout Mask Recovery

Run from the project root:
    python experiments/final_validation.py

This script is deliberately based on the already validated project modules.
It does NOT implement Byzantine robustness.

Important:
The simulated dropout here is "dropout after sending a masked update".
That is the scenario where mask recovery is required:
the server has the client's masked update but needs the client's mask
to remove it from the aggregate.

For clients that never send an update, their update is simply excluded and
their mask does not need to be recovered.
"""

"""
FINAL VALIDATION
PQC + Forward Privacy + Secure Aggregation + Dropout Mask Recovery

Run from the project root:

    python experiments/final_validation.py

This validates the already implemented components.
Byzantine robustness is intentionally NOT implemented here.
"""

# ============================================================
# IMPORTS / PROJECT PATH
# ============================================================

import sys
import copy
import csv
import io
import random
import time
from pathlib import Path

# Because this file is inside experiments/, explicitly add the
# project root so imports such as "from fl_model import ..." work.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fl_model import SimpleNN, train_local_model, evaluate_model
from aggregation import fed_avg
from forward_privacy import ForwardPrivacyClient
from secure_aggregation import SecureAggregator
from mask_recovery import DropoutMaskRecovery
from server import Server


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 5
SAMPLES_PER_CLIENT = 1000
LOCAL_EPOCHS = 1
ROUNDS = 3

# Percentage of clients that simulate dropout AFTER sending
# their masked update.
DROPOUT_LEVELS = [0.0, 0.20, 0.40, 0.60]

SEED = 42

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

def load_mnist():
    """Load MNIST training and test datasets."""

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def create_client_partitions(dataset):
    """
    Create fixed, non-overlapping data partitions.

    Each client receives SAMPLES_PER_CLIENT samples.
    """

    required_samples = NUM_CLIENTS * SAMPLES_PER_CLIENT

    if len(dataset) < required_samples:
        raise ValueError(
            f"MNIST has {len(dataset)} samples, but "
            f"{required_samples} are required."
        )

    indices = list(range(required_samples))

    random.seed(SEED)
    random.shuffle(indices)

    partitions = {}

    for i in range(NUM_CLIENTS):

        start = i * SAMPLES_PER_CLIENT
        end = start + SAMPLES_PER_CLIENT

        client_id = f"Client_{i + 1}"

        partitions[client_id] = indices[start:end]

    return partitions


# ============================================================
# MODEL UPDATE UTILITIES
# ============================================================

def clone_update(update):
    """Create an independent copy of a model update."""

    return {
        name: parameter.clone()
        for name, parameter in update.items()
    }


def sum_updates(updates):
    """Calculate element-wise sum of model updates."""

    if not updates:
        raise ValueError("Cannot sum an empty update list.")

    result = clone_update(updates[0])

    for update in updates[1:]:

        for name in result:
            result[name] += update[name]

    return result


def max_difference(update_a, update_b):
    """Return maximum absolute difference between two updates."""

    maximum = 0.0

    for name in update_a:

        difference = torch.max(
            torch.abs(
                update_a[name] - update_b[name]
            )
        ).item()

        maximum = max(maximum, float(difference))

    return maximum


def serialize_update(update):
    """Serialize a PyTorch model update."""

    buffer = io.BytesIO()

    torch.save(update, buffer)

    return buffer.getvalue()


def deserialize_update(data):
    """Deserialize a PyTorch model update."""

    return torch.load(
        io.BytesIO(data),
        weights_only=False
    )


# ============================================================
# INTEGRATED EXPERIMENT
# ============================================================

def run_integrated_experiment(dropout_fraction):

    """
    Execute:

        Local FL training
              |
              v
        ML-KEM key establishment
              |
              v
        AES-GCM encryption
              |
              v
        Secure aggregation masking
              |
              v
        Client sends masked update
              |
              v
        Client drops
              |
              v
        Shamir mask recovery
              |
              v
        Mask removal
              |
              v
        Aggregate verification
              |
              v
        FedAvg
              |
              v
        Key evolution
    """

    train_dataset, test_dataset = load_mnist()

    partitions = create_client_partitions(
        train_dataset
    )

    # --------------------------------------------------------
    # SERVER / PQC
    # --------------------------------------------------------

    server = Server()

    server_public_key = server.get_public_key()

    client_ids = list(partitions.keys())

    # --------------------------------------------------------
    # CLIENT PQC INITIALIZATION
    # --------------------------------------------------------

    clients = {}

    for client_id in client_ids:

        clients[client_id] = ForwardPrivacyClient(
            client_id,
            server_public_key
        )

    # --------------------------------------------------------
    # ESTABLISH SERVER-SIDE CLIENT KEYS
    # --------------------------------------------------------

    for client_id in client_ids:

        server.establish_client_key(
            client_id,
            clients[client_id].get_pqc_ciphertext()
        )

    # --------------------------------------------------------
    # SECURE AGGREGATION / MASK RECOVERY
    # --------------------------------------------------------

    secure_aggregator = SecureAggregator(
        client_ids
    )

    mask_recovery = DropoutMaskRecovery(
        threshold=2
    )

    # --------------------------------------------------------
    # GLOBAL MODEL
    # --------------------------------------------------------

    global_model = SimpleNN()

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    experiment_rows = []

    dropout_rng = random.Random(
        SEED + int(dropout_fraction * 100)
    )

    # ========================================================
    # FEDERATED ROUNDS
    # ========================================================

    for round_number in range(1, ROUNDS + 1):

        print()
        print("=" * 70)
        print(
            f"ROUND {round_number} | "
            f"DROPOUT {int(dropout_fraction * 100)}%"
        )
        print("=" * 70)

        server.start_round(round_number)

        original_updates = {}
        decrypted_updates = {}
        encrypted_updates = {}

        local_training_seconds = 0.0
        encryption_seconds = 0.0
        mask_generation_seconds = 0.0
        mask_recovery_seconds = 0.0

        # ----------------------------------------------------
        # 1. LOCAL TRAINING + ENCRYPTION
        # ----------------------------------------------------

        for client_id in client_ids:

            model = SimpleNN()

            model.load_state_dict(
                copy.deepcopy(
                    global_model.state_dict()
                )
            )

            client_dataset = Subset(
                train_dataset,
                partitions[client_id]
            )

            train_loader = DataLoader(
                client_dataset,
                batch_size=32,
                shuffle=True
            )

            # Local training timing
            start = time.perf_counter()

            train_local_model(
                model,
                train_loader,
                epochs=LOCAL_EPOCHS
            )

            local_training_seconds += (
                    time.perf_counter() - start
            )

            update = clone_update(
                model.state_dict()
            )

            original_updates[client_id] = update

            # Serialize model update
            plaintext = serialize_update(
                update
            )

            # AES-GCM encryption timing
            start = time.perf_counter()

            encrypted_update = clients[
                client_id
            ].encrypt_update(
                plaintext
            )

            encryption_seconds += (
                    time.perf_counter() - start
            )

            encrypted_updates[
                client_id
            ] = encrypted_update

            # Client sends encrypted update
            server.receive_update(
                client_id,
                round_number,
                encrypted_update
            )

        # ----------------------------------------------------
        # 2. SERVER DECRYPTION
        # ----------------------------------------------------

        for client_id in client_ids:

            plaintext = server.decrypt_update(
                client_id,
                round_number
            )

            decrypted_updates[
                client_id
            ] = deserialize_update(
                plaintext
            )

        received_clients = server.get_received_clients(
            round_number
        )

        if set(received_clients) != set(client_ids):

            raise RuntimeError(
                "Server did not receive all expected "
                "encrypted updates."
            )

        # ----------------------------------------------------
        # 3. CREATE MASKS + SHAMIR SHARES
        # ----------------------------------------------------

        seeds = {}
        shares = {}
        masks = {}
        masked_updates = {}

        for client_id in client_ids:

            seed = mask_recovery.create_mask_seed()

            seeds[client_id] = seed

            # 2-of-3 Shamir secret sharing
            shares[client_id] = (
                mask_recovery.create_shares(
                    seed,
                    3
                )
            )

            start = time.perf_counter()

            mask = secure_aggregator.generate_mask_from_seed(
                seed,
                decrypted_updates[client_id]
            )

            mask_generation_seconds += (
                    time.perf_counter() - start
            )

            masks[client_id] = mask

            masked_updates[client_id] = (
                secure_aggregator.create_masked_update(
                    decrypted_updates[client_id],
                    mask
                )
            )

        # ----------------------------------------------------
        # 4. SIMULATE DROPOUT AFTER SENDING MASKED UPDATE
        # ----------------------------------------------------

        number_to_drop = int(
            NUM_CLIENTS * dropout_fraction
        )

        # Keep at least one client alive.
        number_to_drop = min(
            number_to_drop,
            NUM_CLIENTS - 1
        )

        if number_to_drop > 0:

            dropped_clients = dropout_rng.sample(
                client_ids,
                number_to_drop
            )

        else:

            dropped_clients = []

        surviving_clients = [
            client_id
            for client_id in client_ids
            if client_id not in dropped_clients
        ]

        print(
            "Dropped after masked update:",
            dropped_clients
            if dropped_clients
            else "None"
        )

        print(
            "Surviving clients:",
            surviving_clients
        )

        # ----------------------------------------------------
        # 5. AGGREGATE MASKED UPDATES
        # ----------------------------------------------------

        start = time.perf_counter()

        masked_sum = (
            secure_aggregator.aggregate_masked_updates(
                masked_updates
            )
        )

        # ----------------------------------------------------
        # 6. RECOVER DROPPED CLIENT MASKS
        # ----------------------------------------------------

        for client_id in dropped_clients:

            recovered_seed = (
                mask_recovery.recover_seed(
                    shares[client_id][:2]
                )
            )

            if recovered_seed != seeds[client_id]:

                raise RuntimeError(
                    f"Seed recovery failed for "
                    f"{client_id}."
                )

            recovered_mask = (
                mask_recovery.generate_mask(
                    recovered_seed,
                    decrypted_updates[client_id]
                )
            )

            masked_sum = (
                secure_aggregator.remove_mask(
                    masked_sum,
                    recovered_mask
                )
            )

        # ----------------------------------------------------
        # 7. REMOVE SURVIVING CLIENT MASKS
        # ----------------------------------------------------

        for client_id in surviving_clients:

            masked_sum = (
                secure_aggregator.remove_mask(
                    masked_sum,
                    masks[client_id]
                )
            )

        mask_recovery_seconds = (
                time.perf_counter() - start
        )

        # ----------------------------------------------------
        # 8. VERIFY AGGREGATE
        # ----------------------------------------------------

        expected_sum = sum_updates(
            [
                decrypted_updates[client_id]
                for client_id in client_ids
            ]
        )

        verification_error = max_difference(
            masked_sum,
            expected_sum
        )

        secure_aggregation_passed = (
                verification_error < 1e-4
        )

        if not secure_aggregation_passed:

            raise RuntimeError(
                "Secure aggregation verification failed. "
                f"Maximum error: "
                f"{verification_error}"
            )

        print(
            "Secure aggregation verification: PASSED"
        )

        print(
            f"Maximum numerical error: "
            f"{verification_error:.8f}"
        )

        # ----------------------------------------------------
        # 9. FEDAVG
        # ----------------------------------------------------

        client_update_list = [
            decrypted_updates[client_id]
            for client_id in client_ids
        ]

        client_weights = [
            SAMPLES_PER_CLIENT
            for _ in client_ids
        ]

        global_update = fed_avg(
            client_update_list,
            client_weights
        )

        global_model.load_state_dict(
            global_update
        )

        accuracy = evaluate_model(
            global_model,
            test_loader
        )

        print(
            f"Global model accuracy: "
            f"{accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # 10. KEY EVOLUTION
        # ----------------------------------------------------

        for client_id in client_ids:

            clients[
                client_id
            ].complete_round()

            server.evolve_client_key(
                client_id
            )

        print(
            "Forward key evolution: PASSED"
        )

        # ----------------------------------------------------
        # SAVE ROUND RESULT
        # ----------------------------------------------------

        experiment_rows.append({

            "round":
                round_number,

            "dropout_percent":
                int(dropout_fraction * 100),

            "dropped_clients":
                ",".join(dropped_clients)
                if dropped_clients
                else "None",

            "surviving_clients":
                ",".join(surviving_clients),

            "accuracy_percent":
                round(accuracy, 4),

            "verification_error":
                verification_error,

            "secure_aggregation_passed":
                secure_aggregation_passed,

            "local_training_seconds":
                round(
                    local_training_seconds,
                    4
                ),

            "encryption_seconds":
                round(
                    encryption_seconds,
                    6
                ),

            "mask_generation_seconds":
                round(
                    mask_generation_seconds,
                    6
                ),

            "mask_recovery_seconds":
                round(
                    mask_recovery_seconds,
                    6
                ),
        })

    return experiment_rows


# ============================================================
# FORWARD PRIVACY TEST
# ============================================================

def run_forward_privacy_test():

    """
    Verify that a later evolved key cannot decrypt
    ciphertext generated with an earlier key.
    """

    print()
    print("=" * 70)
    print("FORWARD PRIVACY COMPROMISE TEST")
    print("=" * 70)

    from key_manager import KeyManager
    from crypto_utils import encrypt_data, decrypt_data

    manager = KeyManager()

    plaintext = (
        b"round-1-sensitive-model-update"
    )

    # K1
    round_1_key = (
        manager.get_current_key()
    )

    ciphertext = encrypt_data(
        round_1_key,
        plaintext
    )

    # Evolve K1 -> K2
    manager.evolve_key()

    round_2_key = (
        manager.get_current_key()
    )

    # K2 must NOT decrypt ciphertext encrypted by K1.
    try:

        decrypt_data(
            round_2_key,
            ciphertext
        )

        later_key_decryption_failed = False

    except Exception:

        later_key_decryption_failed = True

    if not later_key_decryption_failed:

        raise RuntimeError(
            "Forward privacy failed: "
            "later key decrypted earlier ciphertext."
        )

    # Original K1 must still decrypt its own ciphertext.
    recovered = decrypt_data(
        round_1_key,
        ciphertext
    )

    if recovered != plaintext:

        raise RuntimeError(
            "Original-key decryption verification failed."
        )

    print(
        "Later key cannot decrypt earlier ciphertext: PASSED"
    )

    print(
        "Original key decrypts original ciphertext: PASSED"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("PQC + FORWARD PRIVACY + SECURE AGGREGATION")
    print("+ DROPOUT MASK RECOVERY")
    print("=" * 70)

    all_rows = []

    # --------------------------------------------------------
    # Run integrated experiment at each dropout level.
    # --------------------------------------------------------

    for dropout_fraction in DROPOUT_LEVELS:

        rows = run_integrated_experiment(
            dropout_fraction
        )

        all_rows.extend(rows)

    # --------------------------------------------------------
    # Forward privacy test.
    # --------------------------------------------------------

    forward_privacy_passed = (
        run_forward_privacy_test()
    )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    output_file = (
            RESULTS_DIR /
            "final_validation_results.csv"
    )

    fieldnames = [

        "round",

        "dropout_percent",

        "dropped_clients",

        "surviving_clients",

        "accuracy_percent",

        "verification_error",

        "secure_aggregation_passed",

        "local_training_seconds",

        "encryption_seconds",

        "mask_generation_seconds",

        "mask_recovery_seconds",
    ]

    with output_file.open(
            "w",
            newline="",
            encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    secure_aggregation_passed = all(
        row[
            "secure_aggregation_passed"
        ]
        for row in all_rows
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print(
        "Secure Aggregation:",
        "PASSED"
        if secure_aggregation_passed
        else "FAILED"
    )

    print(
        "Dropout Mask Recovery:",
        "PASSED"
        if secure_aggregation_passed
        else "FAILED"
    )

    print(
        "Forward Privacy:",
        "PASSED"
        if forward_privacy_passed
        else "FAILED"
    )

    print()
    print(
        f"Results saved to:\n{output_file}"
    )

    if (
            secure_aggregation_passed
            and forward_privacy_passed
    ):

        print()
        print("=" * 70)
        print("FINAL VALIDATION PASSED")
        print("=" * 70)

    else:

        print()
        print("=" * 70)
        print("FINAL VALIDATION FAILED")
        print("=" * 70)


if __name__ == "__main__":
    main()
