"""
This module contains the code to choose which experiment to execute.
"""

# Standard libraries
from typing import Literal

# Own module
from src.explain.experiments import examples, full_tables, drop_tables


def main() -> None:
    """
    This is the main function that executes teh experiment that you
    define with the experiment_name variable.

    Returns:
        None.
    """

    # Define experiment to execute
    experiment_name: Literal["examples", "full_tables", "drop_tables"] = "examples"

    match experiment_name:
        case "examples":
            examples.main()
        case "full_tables":
            full_tables.main()
        case "drop_tables":
            drop_tables.main()

    return None


if __name__ == "__main__":
    main()
