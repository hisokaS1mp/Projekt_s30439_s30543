import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig
from asi_project.data import assert_valid


def validate_data(
    df: pd.DataFrame,
    contract: DatasetContract,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Sprawdza kontrakt i przekazuje poprawne dane do kolejnego node'a."""
    assert_valid(df, contract, config)
    return df
