"""Help-only command stubs.

Later milestones replace the refusal in each script. M0.0 accepts ``--help``
and refuses every other invocation, so a stub cannot download data or train.
"""

import argparse


def unimplemented(prog: str, description: str, milestone: str, planned: str) -> None:
    """Parse ``--help`` and refuse every other invocation."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=(
            f"Planned for milestone {milestone}. Not accepted in M0.0:\n"
            f"  {planned}\n\n"
            "M0.0 serves --help and refuses every other invocation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()
    parser.error(
        f"{prog} is not implemented until milestone {milestone}; refusing to run."
    )
