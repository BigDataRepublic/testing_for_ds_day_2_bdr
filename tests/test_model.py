from pathlib import Path
from typing import Iterator

import pytest
from sklearn.model_selection import GridSearchCV

from dishwashers.data import get_prediction_data, get_trainings_data
from dishwashers.model import DishwasherModel

# -- Exercise 4 --
# Write a module scoped fixture that yields an instance of DishwasherModel, trained on the labelled data in
# dummy_dishwasher_registration.csv. Make it depend on the fixture restrict_grid_search().


@pytest.fixture(scope="module")
def dishwasher_model(restrict_grid_search: GridSearchCV) -> Iterator[DishwasherModel]:
    df_input = get_trainings_data(
        file_path=Path("tests/test_data/dummy_dishwasher_registration.csv")
    )

    model = DishwasherModel()
    model.train(data=df_input)

    yield model


# -- Exercise 5 --
# Write a test that uses the fixture you just created to test that all predictions on the unlabeled data in
# dummy_dishwasher_registration.csv are positive.
def test_dishwasher_model(dishwasher_model: DishwasherModel) -> None:
    df_input = get_prediction_data(
        file_path=Path("tests/test_data/dummy_dishwasher_registration.csv")
    )

    df_output = dishwasher_model.predict(data=df_input)

    assert all(df_output["prediction_dishwashers"] > 0)
