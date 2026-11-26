# ADR-2: Który model przekazujemy do API?

- **Status:** proponowany · **Sprint:** S3
- **Autor:** Oleksandra Sarry · **Recenzent:** Wiktoria Wydra

## Porównanie
Zakupem kończy się tylko 15,5% sesji, więc główną metryką jest F1.
F1 pochodzi z W&B, a czas dla całej partii i rozmiar z [benchmarku w PR #20](https://github.com/hisokaS1mp/Projekt_s30439_s30543/pull/20).

| Model | F1 (W&B) | Czas predykcji partii | Rozmiar pakietu w benchmarku | RAM po wczytaniu |
|---|---:|---:|---:|---|
| Regresja logistyczna (baseline) | [0,0728](https://wandb.ai/asi-s30439-s30543/online-shoppers/runs/oedhvyns) | 0,0127 s | 0,0075 MiB | brak potwierdzonego pomiaru |
| `ag_fast` (`medium_quality`, 60 s) | [0,4150](https://wandb.ai/asi-s30439-s30543/online-shoppers/runs/wdtvsiia) | 0,0709 s | 290,44 MiB | brak potwierdzonego pomiaru |
| `ag_hq` (`good_quality`, 120 s) | [0,4053](https://wandb.ai/asi-s30439-s30543/online-shoppers/runs/muw9xcxc) | 0,1371 s | 84,88 MiB | brak potwierdzonego pomiaru |

## Decyzja
Wybraliśmy **`ag_hq`**: był wolniejszy od `ag_fast` i miał F1 niższe o 0,01, ale w benchmarku zajmował 85 zamiast 290 MiB.
Regresję odrzuciliśmy, bo przeoczyła większość zakupów. RAM nie rozstrzygnął wyboru — brakuje potwierdzonych pomiarów.

## Konsekwencje
API pobierze `online-shoppers-model:production` z W&B i wczyta `model/` oraz `metadata.json` przez `load_model_package()`.
Opublikowana wersja `v0` waży 213,77 MiB. Smoke test zwrócił 5 poprawnych predykcji bez ponownego treningu.
