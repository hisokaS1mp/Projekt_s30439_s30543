# Model Card

## Model

- AutoGluon Tabular `ag_hq`: preset `good_quality`, limit treningu 120 s.
- [Artefakt W&B `online-shoppers-model:v0`](https://wandb.ai/asi-s30439-s30543/online-shoppers/artifacts/model/online-shoppers-model/v0)
  ma alias `production`. Digest: `6b0a33653decf2e2dfb24d94b3cd683a`. Rozmiar opublikowanego
  pakietu: 213,77 MiB.
- Pakiet opublikowano 25.11.2026 w [runie `vlxthxgb`](https://wandb.ai/asi-s30439-s30543/online-shoppers/runs/vlxthxgb)
  z konta `s30439`.

## Przeznaczenie

- Klasyfikacja binarna: czy trwająca sesja w sklepie zakończy się zakupem. Target: `Revenue`.
  Klasa pozytywna: `true` (kod modelu `1`).
- Model wspiera system analityczny sklepu w decyzji, czy rozważyć akcję marketingową podczas
  sesji. Nie mierzy wpływu takiej akcji na zakup ani nie prognozuje wartości zamówienia.

## Dane

- [UCI Online Shoppers Purchasing Intention](https://doi.org/10.24432/C5F88Q) (`online_shoppers`,
  CC BY 4.0): 12 330 sesji przed filtrowaniem; 15,5% zakończyło się zakupem.
- Po przygotowaniu danych: podział losowy stratyfikowany 80/20 (seed `42`), 9 764 wiersze
  treningowe i 2 441 testowych. Model korzysta z 16 cech; `PageValues` wykluczono zgodnie
  z [kontraktem danych](datasets.md#online_shoppers--online-shoppers-purchasing-intention).
- Fingerprint surowych danych z `metadata.json`: `9fc1ab36ea82aab7170b61fdfd84133f`.

## Metryki

| F1 (główna) | F1 DummyClassifier | Precision | Recall | Accuracy |
|---:|---:|---:|---:|---:|
| 0,4053 | 0,0000 | 0,3158 | 0,5654 | 0,7403 |

Macierz pomyłek na zbiorze testowym: TN = 1 591, FP = 468, FN = 166, TP = 216.

## Ograniczenia i ryzyka

- Klasy są niezbalansowane. Na zbiorze testowym model przeoczył 166 z 382 zakupów,
  a większość sesji, dla których przewidział zakup, nie zakończyła się zakupem (468 z 684).
- Skuteczności nie zweryfikowano w innych sklepach ani na późniejszych danych. Część cech
  wymaga statystyk analitycznych dostępnych przed końcem sesji; formularz użytkownika
  nie jest ich wiarygodnym źródłem.

## Wersje

- Git SHA zapisany w pakiecie: `4fc92153622393c8324d0caf94570ec01464aff6`.
- Python `3.11.17`, AutoGluon Tabular `1.6.3`, LightGBM `4.7.0`, scikit-learn `1.7.2`,
  pandas `2.3.3`, NumPy `2.2.6`, SciPy `1.16.0`, joblib `1.6.0`, CatBoost `1.2.10`.
