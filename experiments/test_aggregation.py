import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from fl_model import SimpleNN
from aggregation import fed_avg


def main():

    print("===================================")
    print("FEDAVG AGGREGATION TEST")
    print("===================================")

    # Create two models
    model_1 = SimpleNN()
    model_2 = SimpleNN()

    # Get their model parameters
    update_1 = {
        name: parameter.clone()
        for name, parameter in model_1.state_dict().items()
    }

    update_2 = {
        name: parameter.clone()
        for name, parameter in model_2.state_dict().items()
    }

    # Aggregate
    averaged_update = fed_avg(
        [update_1, update_2]
    )

    # Verify that parameters exist
    print("\nAggregation successful.")

    print(
        f"Number of parameters: "
        f"{len(averaged_update)}"
    )

    # Check one parameter
    parameter_name = list(
        averaged_update.keys()
    )[0]

    print(
        f"Example parameter: {parameter_name}"
    )

    print(
        f"Parameter shape: "
        f"{averaged_update[parameter_name].shape}"
    )

    print("\nFEDAVG TEST PASSED")


if __name__ == "__main__":
    main()