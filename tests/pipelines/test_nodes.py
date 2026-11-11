import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from asi_project.pipelines.data_science.nodes import evaluate_model, train_model


class NewConfig:
    """Prosty obiekt, który udaje prawdziwy config Kedro,
    żeby train_model miał skąd wziąć nazwy kolumn."""

    numeric_features = ["czas_sesji"]
    categorical_features = ["przegladarka"]


def test_train_model_prosto():
    """Sprawdzenie, czy node buduje i trenuje prawdziwy model."""

    X_train = pd.DataFrame(
        {
            "czas_sesji": [10.5, 20.0, 0.0, 15.5],
            "przegladarka": ["Chrome", "Firefox", "Chrome", "Edge"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1], name="Revenue")

    config = NewConfig()

    model = train_model(X_train, y_train, config)

    assert isinstance(model, Pipeline)

    predykcje = model.predict(X_train)
    assert len(predykcje) == 4


def test_evaluate_model_prosto():
    """Sprawdzenie, czy node ocenia model i zwraca słownik z metrykami."""

    X_test = pd.DataFrame({"kolumna1": [1, 2]})
    y_test = pd.Series([1, 0], name="Revenue")

    prosty_model = Pipeline(steps=[("clf", DummyClassifier(strategy="constant", constant=1))])
    prosty_model.fit(X_test, y_test)

    metryki = evaluate_model(prosty_model, X_test, y_test)

    assert isinstance(metryki, dict)
    assert len(metryki) > 0
