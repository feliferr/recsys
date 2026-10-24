# Notas de pesquisa: estado da arte em RecSys (out/2026)

> **Nível de confiança.** Notas baseadas em títulos e resumos de artigos, não em leitura integral dos
> PDFs. Itens marcados com (não verificado) são conhecimento de base ou extrapolação. Confira nos
> artigos antes de citar números.

## 1. State Space Models (Mamba) vs. Transformers em recomendação sequencial

**Estabelecido**
- Mamba4Rec foi o primeiro trabalho a aplicar SSMs seletivos a recomendação sequencial, com argumento
  de eficiência (complexidade linear no comprimento da sequência) ([arXiv 2403.03900](https://arxiv.org/abs/2403.03900)).
  Seguiram-se SIGMA ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33336/35491)), M2Rec
  multi-escala ([2505.04445](https://arxiv.org/pdf/2505.04445)), STAR-Rec ([2505.03484](https://arxiv.org/pdf/2505.03484))
  e um estudo de sequências longas ([2403.16371](https://arxiv.org/abs/2403.16371)). Mamba-3
  ([2603.15569](https://arxiv.org/abs/2603.15569)) é uma evolução da arquitetura base.
- Há um estudo comparativo de eficiência e desempenho ([2408.05606](https://arxiv.org/pdf/2408.05606)).

**O ponto crítico de 2026**
- *"Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?"* (Petrov et al.):
  duas sondas simples, sem representação sequencial de ordem superior (SeqRules e PCTM, baseadas em
  transições par a par e recência), superam o eSASRec em 15–38% em três datasets Amazon e 4,4% no
  MovieLens-1M, mas ficam 27,3% abaixo no MovieLens-20M ([arXiv 2608.19833](https://arxiv.org/abs/2608.19833);
  [RecSys 2026, sessão 2](https://recsys.acm.org/recsys26/session-2/)).
- **Implicação:** muitos ganhos de "Mamba > Transformer" (ou o contrário) em benchmarks públicos podem
  não medir modelagem sequencial de fato. Antes de qualquer SSM, tenha baselines fortes e baratos
  (popularidade, transições par a par, recência) e um *capability probe*.
- **Onde SSMs têm vantagem plausível:** sequências muito longas (10⁴ a 10⁵ eventos) e inferência
  incremental com estado de tamanho constante, sem KV cache crescente. O lado Transformer responde com
  cache de baixo posto (*SequenceO1*, 100 mil eventos, RecSys 2026; ver também RelayGR abaixo).
- Em sequências curtas (típicas de sessão), SASRec bem treinado tende a empatar com Mamba; o ganho é de
  custo, não de acurácia (não verificado).

**Implicações para este projeto:** o foco atual é conteúdo (item-item) e interações públicas.
Modelos sequenciais só entram quando a avaliação indicar que vale. O `SequenceEncoder` fica como ponto
de extensão (Protocol), sem implementar SSM por enquanto.

## 2. Multimodalidade e LLMs

1. **LLM/MLLM como codificador de item.** Benchmarks: BLaIR ([ACL 2026](https://aclanthology.org/2026.acl-long.147/))
   para LLMs como encoders semânticos; UniRec ([2601.19423](https://arxiv.org/pdf/2601.19423)) com
   encoder multimodal unificado e Q-Former; OneReason ([2606.06260](https://arxiv.org/pdf/2606.06260))
   com tokenizador de item multimodal (ViT e áudio) destilado em embeddings densos. Survey:
   [2505.09777](https://arxiv.org/pdf/2505.09777).
2. **Recomendação generativa com Semantic IDs.** Itens tokenizados em códigos discretos (RQ-VAE etc.) e
   gerados de forma autorregressiva: [Understanding Generative Rec. with Semantic IDs (KDD 2026)](https://arxiv.org/pdf/2509.25522),
   LSIG ([WWW'26](https://dl.acm.org/doi/10.1145/3774904.3792834)), DECOR (SIGIR 2026),
   Differentiable Semantic ID ([ACM](https://dl.acm.org/doi/10.1145/3805712.3809641)). Tensão aberta:
   sinal semântico (conteúdo) vs. sinal colaborativo.
3. **Agentes LLM end-to-end** (planejamento, memória, simulação de usuário), por exemplo
   [2607.07108](https://arxiv.org/pdf/2607.07108). Úteis para simulação e avaliação; ainda caros e de
   latência alta para servir diretamente.

**Implicações para este projeto:** um `ItemEncoder` plugável, que vai do baseline de conteúdo (TF-IDF de
gêneros) a encoders multimodais pré-treinados (texto/sinopse, imagem, áudio), com embeddings versionados
em um feature store. Semantic IDs ficam para depois, condicionados a dados colaborativos.

## 3. Serving de baixa latência: KV cache e multi-tenancy

- **MTServe:** cache KV hierárquico em duas camadas para recomendação generativa, até 3,1× de
  aceleração ([2604.22881](https://arxiv.org/html/2604.22881v1)).
- **RACER / "When KV Meets Embeddings":** tabelas de embedding quentes e KV cache disputam HBM;
  alocação adaptativa com controlador PPO reduz o P99 em 24–38% frente à melhor política estática
  ([2605.04450](https://arxiv.org/abs/2605.04450)).
- **RcLLM:** reuso de blocos KV não-prefixo ([2605.07443](https://arxiv.org/html/2605.07443)).
- **RelayGR:** inferência *relay-race* entre estágios para sequências longas ([2601.01712](https://arxiv.org/pdf/2601.01712)).
- **Unified AI Gateway:** roteamento de modelos e gestão de KV cache ([2609.06940](https://arxiv.org/pdf/2609.06940)).

**Leitura prática:** esses trabalhos miram recomendação generativa/LLM em escala de GPU. Para um
recomendador item-item por embeddings, os ganhos práticos vêm de (a) índice ANN pré-computado
(HNSW, ScaNN ou serviço gerenciado), (b) top-K por item calculado offline quando o catálogo é pequeno,
(c) cache de resposta e (d) multi-tenant via namespace de índice e modelo por tenant, com artefatos
versionados. KV cache só importa com modelo generativo ou sequencial.

## 4. Avaliação offline e simulação com menos viés

Problema central: logs refletem a política anterior (viés de exposição e seleção).
- **Off-policy evaluation (OPE):** IPS, SNIPS, Doubly Robust; Meta Off-Policy Estimation
  ([2508.07914](https://arxiv.org/pdf/2508.07914)); seleção de política em sistemas de dois estágios com
  suporte (CASP, [2604.23022](https://arxiv.org/pdf/2604.23022)); cuidado ao usar nDCG como estimador OPE
  ([2307.15053](https://arxiv.org/pdf/2307.15053)).
- **Protocolo:** estudos de amostragem em avaliação offline simulando vieses de exposição a partir de um
  dataset totalmente observado ([RecSys 2025, sessão 9](https://recsys.acm.org/recsys25/session-9/)).
- **Boas práticas (não verificado):** split temporal global, avaliação *full-ranking* (sem *sampled
  metrics*), métricas além de acurácia (cobertura, diversidade, novidade, popularidade média,
  calibração) e simuladores (RecSim ou usuários LLM) como filtro antes do A/B, nunca como substituto.
- **No projeto:** `recsys eval baselines` implementa o protocolo acima. Para similaridade item-item,
  `evaluation/item_similarity.py` usa *weak labels* (valores compartilhados de uma coluna como gênero,
  elenco ou direção). Usar co-consumo como rótulo é um próximo passo.

## 5. Apple Silicon e MLX para prototipagem local

- MLX (Apple) tem memória unificada e é bom para iteração rápida e inferência/embeddings locais; o
  ecossistema está em crescimento ([resumo 2026](https://www.digitalapplied.com/blog/apple-mlx-framework-local-ai-developers-2026-guide)).
- PyTorch MPS: histórico de limites (tensor único de ~4 GB reportado em [2511.05502](https://arxiv.org/pdf/2511.05502)
  e em blogs); o PyTorch 2.13 (jul/2026) trouxe FlexAttention ao MPS (fonte secundária, não verificada).
- **Decisão de design:** treino em PyTorch (portável entre MPS, CUDA e nuvem); extração de embeddings
  atrás de um `Backend` com implementação MLX opcional. MLX não é dependência obrigatória (o CI roda em
  Linux).
