from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    """Describes the columns needed to turn a panel into a model dataset."""

    entity_column: str
    time_column: str
    target_column: str
    feature_columns: Sequence[str]
    positive_label: Any = 1
    categorical_columns: Sequence[str] = ()
