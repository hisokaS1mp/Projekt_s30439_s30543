# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart TD
    DATA["UCI Online Shoppers"]
    CONTRACT["Kontrakt danych<br/>datasets.yaml + project.yaml"]

    TRAIN["Przygotowanie i trening - S1<br/>normalizacja wejścia → filtrowanie tylko treningowe<br/>split → Pipeline: preprocessing + model<br/>docelowo Kedro - S2"]
    MODEL["Pakiet modelu - S3<br/>dopasowany Pipeline"]
    API["API - S4<br/>walidacja i ta sama normalizacja<br/>predykcja wczytanym modelem"]
    UI["UI - S5<br/>analityk sklepu"]
    DB[("Persistence - S4-S6<br/>zapis predykcji, Postgres")]
    CLOUD["Cloud Run - S6<br/>wdrożenie API i UI"]

    DATA --> TRAIN
    CONTRACT -.-> TRAIN
    CONTRACT -.-> API
    TRAIN --> MODEL
    MODEL -->|"wczytanie przy starcie"| API
    UI -->|"dane sesji"| API
    API -->|"klasa i prawdopodobieństwo zakupu"| UI
    API --> DB
    CLOUD -.-> API
    CLOUD -.-> UI
```

W S1 korzystamy z danych UCI Online Shoppers zgodnie z kontraktem w datasets.yaml i project.yaml. Normalizacja działa zarówno podczas przygotowania danych, jak i predykcji, filtrowanie rekordów jest wykonywane tylko podczas przygotowania danych do treningu, a preprocessing modelu jest dopasowywany wyłącznie na train. Planowane API wczyta zapisany pipeline, zwróci analitykowi klasę i prawdopodobieństwo zakupu oraz zapisze predykcję w bazie. Elementy S2-S6 są planowane, w tym automatyzacja treningu w Kedro i wdrożenie API oraz UI w Cloud Run.
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
