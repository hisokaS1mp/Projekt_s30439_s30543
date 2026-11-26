import argparse
import json
import os
import sys

from asi_project.model_smoke import smoke_production_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pobiera model production z W&B i sprawdza predykcje dla pięciu wierszy."
    )
    entity = os.environ.get("WANDB_ENTITY")
    parser.add_argument("--entity", default=entity, required=not entity, help="nazwa zespołu w W&B")
    args = parser.parse_args(argv)

    try:
        result = smoke_production_model(entity=args.entity)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Smoke test modelu nie przeszedł: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
