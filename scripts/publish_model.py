import argparse
import os
import sys
from pathlib import Path

from asi_project.model_artifact import publish_production_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publikuje pakiet modelu w W&B z aliasem production."
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("data/06_models/production_package"),
        help="katalog pakietu z modelem i metadata.json",
    )
    entity = os.environ.get("WANDB_ENTITY")
    parser.add_argument("--entity", default=entity, required=not entity, help="nazwa zespołu w W&B")
    args = parser.parse_args(argv)

    try:
        version = publish_production_model(args.package, entity=args.entity)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Nie udało się opublikować modelu: {error}", file=sys.stderr)
        return 1
    print(f"production -> {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
