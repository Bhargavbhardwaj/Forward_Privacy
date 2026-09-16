import copy
import io

import torch
from torch.utils.data import DataLoader, Subset

from fl_model import SimpleNN, train_local_model
from forward_privacy import ForwardPrivacyClient


class FLClient:

    def __init__(
            self,
            client_id,
            dataset,
            indices,
            server_public_key
    ):

        self.client_id = client_id

        self.dataset = Subset(
            dataset,
            indices
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True
        )

        self.model = SimpleNN()

        # PQC + Forward Privacy
        self.forward_privacy = ForwardPrivacyClient(
            client_id,
            server_public_key
        )

    def train(self, global_model):

        self.model.load_state_dict(
            copy.deepcopy(
                global_model.state_dict()
            )
        )

        train_local_model(
            self.model,
            self.loader,
            epochs=1
        )

        return self.model

    def get_model_update(self):

        update = {}

        for name, parameter in self.model.state_dict().items():

            update[name] = parameter.clone()

        return update

    def encrypt_update(self, update):

        buffer = io.BytesIO()

        torch.save(
            update,
            buffer
        )

        plaintext = buffer.getvalue()

        encrypted_update = (
            self.forward_privacy.encrypt_update(
                plaintext
            )
        )

        return encrypted_update

    def complete_round(self):

        self.forward_privacy.complete_round()

    def get_current_key(self):

        return self.forward_privacy.get_current_key()

    def get_pqc_ciphertext(self):

        return self.forward_privacy.get_pqc_ciphertext()