"""
This module contains the code to choose which experiment to execute.
"""

# Standard libraries
from typing import Literal

# Own module
from src.explain.experiments import examples, dropout, full


def main() -> None:
    """
    This is the main function that executes teh experiment that you
    define with the experiment_name variable.

    Returns:
        None.
    """

    # Define experiment to execute
    experiment_name: Literal["examples", "dropout", "full"] = "dropout"

    match experiment_name:
        case "examples":
            examples.main()
        case "dropout":
            dropout.main()
        case "full":
            full.main()

    return None


if __name__ == "__main__":
    main()
