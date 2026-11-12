# ADR-1: Podział logiki przetwarzania danych

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Wiktoria Wydra
- **Recenzent:** Oleksandra Sarry

## Kontekst

W ramach Sprintu 2 przenosimy proces treningowy z notatnika do frameworka Kedro dla zbioru `online_shoppers`. Naszym celem jest zapewnienie pełnej reprodukowalności (uruchamianie przez `kedro run`) bez utraty elastyczności. Ograniczeniem jest ryzyko wycieku danych (data leakage) przy podziale na zbiór treningowy i testowy oraz konieczność przygotowania kodu w sposób, który umożliwi w przyszłości łatwe serwowanie predykcji dla pojedynczych rekordów.

## Rozważone opcje

1. **Opcja A (Wszystko w Kedro)** — Przeniesienie całej logiki czyszczenia, skalowania, imputacji i treningu jako oddzielne, niezależne węzły (nodes) w Kedro.
2. **Opcja B (Hybrydowa)** — Podział na trzy obszary: *input normalization* (wspólne, w `data.py`), *training-only filtering* (tylko w Kedro jako node) oraz *model preprocessing* (w obiekcie `sklearn.Pipeline`).

## Decyzja

Wybieramy **Opcję B**, ponieważ jasno oddziela transformacje aplikowalne do pojedynczego klienta na produkcji od operacji stricte treningowych. Zamknięcie preprocessingu modelu w `sklearn.Pipeline` to jedyny bezpieczny sposób na uniknięcie wycieku statystyk (np. średnich dla imputacji) ze zbioru testowego do treningowego, co mogłoby sztucznie zaburzyć ewaluację modelu.

| Wielkość | Wartość | Skąd pochodzi |
|---|---|
| Liczba odrzucanych duplikatów (training-only) | ~125 | Analiza z EDA (`01_eda.ipynb`) |
| Rozmiar próbki testowej (`test_size`) | 0.2 (20%) | Plik konfiguracyjny `parameters.yml` |

## Odrzucone alternatywy

Odrzuciliśmy **Opcję A** (wyciągnięcie m.in. skalera i imputacji do osobnych nodów w Kedro przed podziałem zbioru). Zastosowanie przekształceń na całej ramce danych wejściowych przed wykonaniem `train_test_split` skutkowałoby włączeniem wiedzy o zbiorze testowym w parametry skalera, czyli wyciekiem danych. Wymusiłoby to również przekazywanie wyuczonych obiektów scikit-learn pomiędzy wieloma nodami, co psuje enkapsulację.

## Konsekwencje

* **Co ułatwia:** Znacznie upraszcza wdrażanie modelu na produkcję (inference). Aby obsłużyć nowego klienta API, wystarczy przepuścić dane przez wspólną funkcję `input normalization` w `data.py` i przekazać prosto do zapisanej paczki z `sklearn.Pipeline`.
* **Co utrudnia:** Wymaga od członków zespołu ciągłej czujności architektonicznej — musimy świadomie decydować, czy nowa transformacja to globalna modyfikacja wejścia (Kedro / `data.py`), czy tylko element uczenia maszyny (scikit-learn).
* **Kolejne sprinty:** Przygotowana struktura ułatwi integrację z AutoGluon w Sprincie 3.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-07 | utworzenie |
