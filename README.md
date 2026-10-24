# recsys

Toolkit de sistemas de recomendação (conteúdo e sequencial) com foco em **avaliação offline rigorosa antes
de modelos treinados**: contratos de dados tipados, splits temporais, ranking completo e baselines
reproduzíveis. Modelos treinados e serving estão no [roadmap](docs/roadmap.md).

*English summary: an offline-first recommender-systems toolkit with typed data contracts, temporal
splits, full-ranking evaluation (ranking and beyond-accuracy metrics) and non-learned baselines.
Learned models and serving are planned.*

## Estado atual

| Área | Situação |
|---|---|
| Contratos de dados (Pydantic) e layout Parquet | Pronto |
| Fontes de dados: sintética determinística, MovieLens, Parquet local | Pronto |
| Splits: leave-last-out e temporal global | Pronto |
| Métricas: nDCG, Recall, HitRate, MAP, cobertura, diversidade, novidade, popularidade média | Pronto |
| Baselines: random, popularidade, conteúdo (TF-IDF de gêneros), transição par a par | Pronto |
| CLI, CI (ruff, mypy, import-linter, pytest em Python 3.11 e 3.12) | Pronto |
| Encoders e feature store, treino em PyTorch, serving | [Planejado](docs/roadmap.md) |

## Como rodar

```bash
conda env create -f environment.yml     # cria o ambiente "recsys" (Python 3.11)
conda activate recsys
ruff check . && ruff format --check .
mypy
lint-imports
pytest
```

Apple Silicon (PyTorch MPS e MLX opcional): `conda env create -f environment-apple-silicon.yml`.
Para atualizar após mudar o `pyproject.toml`: `conda env update -f environment.yml --prune`.

## Dados

```bash
recsys ingest synthetic --out data/synthetic            # dataset sintético determinístico
recsys ingest movielens --src ~/Downloads/ml-latest-small --out data/movielens
```

Layout canônico: `items.parquet` + `interactions.parquet` (contratos em `src/recsys/core/schemas.py`).

## Avaliação offline

```bash
recsys eval baselines --data data/synthetic --out reports/baselines            # leave-last-out
recsys eval baselines --data data/movielens --split temporal --k 10 --k 20
```

Protocolo: split temporal sem vazamento, ranking completo (sem amostragem de negativos) e itens já vistos
excluídos. A saída fica em `results.json` e `results.md`.

Exemplo no dataset sintético (200 itens, 100 usuários, leave-last-out). Esses dados só validam o
pipeline; os números não dizem nada sobre qualidade de recomendação:

| model | n_users | ndcg@10 | recall@10 | coverage@10 | diversity@10 | novelty@10 |
|---|---|---|---|---|---|---|
| random | 100 | 0.0212 | 0.0500 | 1.0000 | 0.8566 | 7.8127 |
| popularity | 100 | 0.0162 | 0.0400 | 0.0850 | 0.7816 | 6.7891 |
| content_genre_tfidf | 100 | 0.1109 | 0.2500 | 0.9400 | 0.2469 | 7.9665 |
| transition | 100 | 0.0523 | 0.0900 | 0.9600 | 0.8221 | 7.4501 |

## Estrutura

```
src/recsys/
  core/         schemas e contratos de dados
  ingestion/    fontes de dados e validação
  evaluation/   splits, métricas, baselines, relatório
  cli.py        recsys ingest | eval
tests/          unit/ e integration/
docs/           architecture.md, roadmap.md, research-notes.md
```

## Documentação

- [Arquitetura](docs/architecture.md): princípios, módulos, contratos e CI.
- [Roadmap](docs/roadmap.md): fases, status e pontos em aberto.
- [Notas de pesquisa](docs/research-notes.md): leituras que motivam as escolhas de design.
