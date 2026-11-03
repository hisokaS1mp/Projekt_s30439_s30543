from dataclasses import replace
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, MemoryDataset

from asi_project.config import get_contract, load_project_config


class ProjectHooks:
    """Łączy deklaracje cech i kontrakt z katalogiem Kedro."""

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        parameters: dict[str, Any],
    ) -> None:
        """Udostępnia konfigurację z parametrami splitu używanymi w tym uruchomieniu."""
        split = parameters["parameters"]["split"]
        config = replace(
            load_project_config(),
            test_size=split["test_size"],
            random_state=split["random_state"],
        )
        catalog["project_config"] = MemoryDataset(config)
        catalog["contract"] = MemoryDataset(get_contract(config))
