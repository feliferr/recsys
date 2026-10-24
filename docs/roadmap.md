# Roadmap

Cada fase é um PR independente, mergeável e testado, sem grandes reescritas de uma vez só.

| Fase | Entrega | Critério de pronto | Status |
|---|---|---|---|
| 0. Fundação | `pyproject.toml`, `environment.yml` (conda), estrutura `src/`, ruff, mypy, pytest, CI | CI verde com teste de fumaça | Concluída |
| 1. Contratos e dados | Schemas Pydantic, `ItemMetadata`, `DataSource` (sintética, MovieLens, Parquet local), validação | Testes unitários; ingestão gera Parquet canônico | Concluída (fonte BigQuery fica para a fase 6) |
| 2. Avaliação primeiro | Splits (leave-last-out, temporal global), métricas de ranking e de cobertura, diversidade, novidade e popularidade, baselines (random, popularidade, conteúdo, transição), *weak labels* para similaridade item-item | Relatório reproduzível via `recsys eval baselines` | Concluída |
| 3. Feature store e extração | Protocol `ItemEncoder`, escrita versionada de embeddings, extração local (MPS, MLX opcional) com encoders pré-treinados de texto, imagem e áudio | Embeddings versionados e teste de reprodutibilidade | Não iniciada |
| 4. Treino em PyTorch | Modelo de ranking parametrizável (two-tower e/ou SASRec, a definir), loop independente de device, tracking, `safetensors` | Supera os baselines da fase 2 na mesma avaliação | Não iniciada |
| 5. Serving | Índice ANN, top-K pré-calculado, FastAPI, resposta tipada, testes de contrato | Latência p99 medida em carga sintética; teste e2e | Não iniciada |
| 6. Nuvem | Containers, fontes BigQuery/GCS, job de treino gerenciado, workflows de deploy sem chaves de longa duração | Um treino rodando na nuvem com a mesma CLI | Não iniciada |
| 7. Experimentos (opcional) | Encoder multimodal pré-treinado vs. baseline de conteúdo; modelos sequenciais (SASRec, Mamba) com *capability probes*; avaliação off-policy | Ganho medido pela avaliação offline | Não iniciada |

## Decisões de projeto

- **Ambiente:** conda (`environment.yml`); o `pyproject.toml` é a fonte de verdade das dependências.
- **Ordem:** avaliação antes de treino (fases 2 a 4).
- **Dados do repositório:** apenas públicos (MovieLens) e sintéticos. Nenhum dado de catálogo ou de
  empresa entra nos fixtures nem na documentação.

## Em aberto

- Família de modelos da fase 4: two-tower, SASRec ou ambos.
- Escopo do serving: API FastAPI própria ou contrato com um serviço externo.
- Modelos sequenciais (fase 7) só entram com logs de interação suficientes; o candidato público é o
  MovieLens.
