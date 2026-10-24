# Arquitetura

Este documento descreve como o projeto está organizado, o que já existe e o que está planejado.
O cronograma fica em [roadmap.md](roadmap.md) e as leituras que motivam as escolhas em
[research-notes.md](research-notes.md).

## Princípios

1. **Medir antes de treinar.** Sem uma métrica de ranking e baselines simples, não dá para saber se um
   modelo treinado vale o custo. A avaliação offline vem antes de qualquer modelo aprendido.
2. **Cuidado com objetivos substitutos.** Treinar para classificar gênero, por exemplo, não garante bons
   rankings. O que importa é avaliar o ranking final.
3. **Dividir antes de reamostrar.** Oversampling, augmentation e normalizações ajustadas nos dados vêm
   depois do split. Caso contrário, duplicatas e estatísticas vazam para a validação.
4. **Split temporal e ranking completo.** Sem amostragem de negativos e com itens já vistos excluídos.
5. **Baselines fortes e baratos primeiro.** Popularidade, conteúdo e transições par a par são o piso
   que qualquer modelo aprendido precisa superar (ver [research-notes.md](research-notes.md), §1).
6. **Estágios desacoplados por contratos de dados.** Parquet e esquemas Pydantic entre os estágios,
   não imports cruzados.

## Módulos

```
src/recsys/
  core/          # schemas (Pydantic) e constantes de colunas               [implementado]
  ingestion/     # fontes de dados e validação                              [implementado]
    sources/     #   DataSource (Protocol): sintética, MovieLens, Parquet local
  evaluation/    # splits, métricas de ranking, baselines, relatório        [implementado]
  cli.py         # recsys ingest | eval                                     [implementado]
  featurestore/  # embeddings versionados                                   [planejado]
  training/      # PyTorch: modelos, losses, loop (cpu | mps | cuda)        [planejado]
  serving/       # índice ANN, API FastAPI                                  [planejado]
  backends/      # ArrayBackend: numpy/torch, MLX opcional                  [planejado]
tests/{unit,integration}/
```

## Contratos entre estágios

- `ItemMetadata`: tabela única e tipada de itens (`item_id`, `name`, `description`, `cover_url`,
  `item_type`, `content_rating`, `genres`, `cast`, `directors`, `countries`, `release_year`).
- `interactions`: interações usuário-item com timestamp.
- Layout canônico em disco: `items.parquet` + `interactions.parquet` (`src/recsys/core/schemas.py`).
- Planejado: `ItemEmbeddingTable` (`item_id`, `embedding`, `encoder_name`, `encoder_version`,
  `created_at`) e modelos em `model.safetensors` + `model_card.json` (hash de config, dados e versões).

## Regras de dependência

Impostas por `import-linter` no CI (`pyproject.toml`):

- `recsys.core` não importa `recsys.ingestion` nem `recsys.cli`.
- `recsys.ingestion` não importa `recsys.cli`.
- `recsys.evaluation` não importa `recsys.ingestion` nem `recsys.cli`.

Regras-alvo quando os demais módulos existirem: `serving`, `training` e `evaluation` dependem apenas de
`featurestore` e `core`; ninguém fora de `training` importa de `training`. O serving carrega um
artefato versionado, não código de treino.

## Qualidade

- `mypy --strict`, `ruff` (lint e format) e `pytest` com cobertura (`hypothesis` está disponível nas
  dependências de desenvolvimento).
- Testes unitários e de integração (CLI de ingestão e de avaliação) com dataset sintético determinístico.
- Os fixtures usam apenas dados sintéticos.

## CI

`.github/workflows/ci.yml`, em PRs e pushes na `master`: matriz Python 3.11 e 3.12 com
`ruff check`, `ruff format --check`, `mypy`, `lint-imports` e `pytest -m "not slow"` com cobertura.
O Dependabot roda mensalmente, em lote, e ignora releases com menos de 7 dias.

## Do local à nuvem (planejado)

A mesma CLI (`recsys ingest | train | eval | build-index`) deve ler uma configuração por ambiente
(local ou nuvem). Local: Parquet em disco, device `mps` ou `cpu`, MLX opcional só para extração de
embeddings. Nuvem: BigQuery/GCS como fontes, treino em job gerenciado e índice FAISS/HNSW no serving.
O extra `gcp` do `pyproject.toml` já reserva as dependências.
