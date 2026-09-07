import copy

import torch
from torch.utils.data import DataLoader, Subset

from fl_model import SimpleNN, train_local_model
from forward_privacy import ForwardPrivacyClient


class FLClient:

    def __init__(
            self,
            client_id,
            dataset,
            indices
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

        self.forward_privacy = (
            ForwardPrivacyClient(client_id)
        )

    def train(self, global_model):

        # Copy global model
        self.model.load_state_dict(
            copy.deepcopy(
                global_model.state_dict()
            )
        )

        # Local training
        train_local_model(
            self.model,
            self.loader,
            epochs=1
        )

        return self.model

    def get_model_update(self):

        update = {}

        for name, parameter in (
                self.model.state_dict().items()
        ):

            update[name] = parameter.clone()

        return update