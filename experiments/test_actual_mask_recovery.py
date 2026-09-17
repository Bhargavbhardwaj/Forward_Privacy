import os
import sys

import torch


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

from mask_recovery import DropoutMaskRecovery
from fl_model import SimpleNN


print("===================================")
print("ACTUAL DROPOUT MASK RECOVERY TEST")
print("===================================")
print()


# ==========================================
# CONFIGURATION
# ==========================================

client_ids = [
    "Client_1",
    "Client_2",
    "Client_3"
]

dropped_client = "Client_3"

surviving_clients = [
    "Client_1",
    "Client_2"
]


# ==========================================
# CREATE CLIENT MODELS
# ==========================================

models = {}

for client_id in client_ids:

    model = SimpleNN()

    # Make each client's model slightly different
    with torch.no_grad():

        for parameter in model.parameters():

            parameter.add_(
                torch.randn_like(parameter) * 0.01
            )

    models[client_id] = model.state_dict()


print("Client updates created.")


# ==========================================
# REFERENCE UPDATE
# ==========================================

reference_update = models["Client_1"]


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

for client_id in client_ids:

    mask_seeds[client_id] = (
        mask_recovery.create_mask_seed()
    )


print("Mask seeds generated.")


# ==========================================
# GENERATE MASKS
# ==========================================

masks = {}

for client_id in client_ids:

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

for client_id in client_ids:

    recovery_shares[client_id] = (
        mask_recovery.create_shares(
            mask_seeds[client_id],
            num_shares=3
        )
    )


print("Recovery shares created.")


# ==========================================
# MASK CLIENT UPDATES
# ==========================================

masked_updates = {}

for client_id in client_ids:

    masked_updates[client_id] = {}

    for name in models[client_id]:

        masked_updates[client_id][name] = (
                models[client_id][name]
                + masks[client_id][name]
        )


print("All client updates masked.")
print()


# ==========================================
# CLIENT 3 DROPS AFTER SENDING UPDATE
# ==========================================

print(
    f"{dropped_client} dropped "
    "after sending masked update."
)

print()


# ==========================================
# CREATE MASKED AGGREGATE
# ==========================================

aggregate = {}

for name in reference_update:

    aggregate[name] = torch.zeros_like(
        reference_update[name]
    )


for client_id in client_ids:

    for name in aggregate:

        aggregate[name] += (
            masked_updates[client_id][name]
        )


print("Masked aggregate created.")


# ==========================================
# REMOVE SURVIVING CLIENT MASKS
# ==========================================

for client_id in surviving_clients:

    for name in aggregate:

        aggregate[name] -= (
            masks[client_id][name]
        )


print("Surviving client masks removed.")


# ==========================================
# RECOVER DROPPED CLIENT SEED
# ==========================================

shares = recovery_shares[dropped_client]

# 2 shares are enough because
# threshold = 2
recovery_shares_for_client = [
    shares[0],
    shares[1]
]

recovered_seed = (
    mask_recovery.recover_seed(
        recovery_shares_for_client
    )
)


print("Dropped client's mask seed recovered.")


# ==========================================
# REGENERATE DROPPED CLIENT MASK
# ==========================================

recovered_mask = (
    mask_recovery.generate_mask(
        recovered_seed,
        reference_update
    )
)


print("Dropped client's mask regenerated.")


# ==========================================
# REMOVE DROPPED CLIENT MASK
# ==========================================

for name in aggregate:

    aggregate[name] -= (
        recovered_mask[name]
    )


print("Recovered dropped client's mask removed.")
print()


# ==========================================
# CALCULATE EXPECTED RESULT
# ==========================================
#
# Client_3 dropped AFTER sending its
# masked update.
#
# Therefore Client_3's actual model
# update is still included in the
# aggregate.
#
# The final result should therefore be:
#
# Client_1 update
# + Client_2 update
# + Client_3 update
#
# Only the masks are removed.
# ==========================================

expected = {}

for name in reference_update:

    expected[name] = torch.zeros_like(
        reference_update[name]
    )


for client_id in client_ids:

    for name in expected:

        expected[name] += (
            models[client_id][name]
        )


# ==========================================
# COMPARE RECOVERED AGGREGATE
# ==========================================

test_passed = True

for name in expected:

    if not torch.allclose(
            aggregate[name],
            expected[name],
            rtol=1e-4,
            atol=1e-4
    ):

        difference = (
                aggregate[name]
                - expected[name]
        ).abs().max().item()

        print(
            f"Mismatch: {name} "
            f"(max difference: {difference})"
        )

        test_passed = False


print()


# ==========================================
# FINAL RESULT
# ==========================================

if test_passed:

    print("===================================")
    print("ACTUAL DROPOUT MASK RECOVERY TEST PASSED")
    print("===================================")

else:

    print("===================================")
    print("ACTUAL DROPOUT MASK RECOVERY TEST FAILED")
    print("===================================")