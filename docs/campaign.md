# Campanha controlada de melhoria

## Referência e orçamento

Baseline auditada: OOF 0.95816557; pública informada 0,95770. Duração 29.12 min. Pico de memória histórico não medido.

Teto autorizado: quatro horas de execução local. Os folds originais permanecem congelados. Triagens usam só o fold 0 e não confirmam nem promovem candidatos.

Critério prático de consistência: cinco folds completos, OOF global acima da baseline, ganho em pelo menos quatro folds e pior delta de fold ≥ -0,0002. Isso não é teste de significância nem previsão de leaderboard.

Amostragem de memória a cada 0,5 s: RSS é memória residente; privada é comprometida no Windows e pode incluir páginas fora da RAM. Picos são amostrais, não limites exatos.

## Custo e prioridade antes das confirmações

Projeções lineares dos perfis de 200 árvores, usadas apenas para alocar orçamento. Overhead, early stopping, profundidade e paginação impedem tratá-las como previsão exata.

| Modelo | Fold 0, 200 árvores (s) | Projeção de CV completa (min) | Pico RSS/privado do perfil (GiB) |
|---|---:|---:|---:|
| lightgbm | 32.6 | 27.7 | 0.58 / 1.43 |
| catboost | 63.5 | 53.2 | 0.62 / 1.80 |
| xgboost | 60.5 | 38.4 | 0.54 / 1.46 |

Ordem de confirmação: LightGBM, CatBoost com 2000 árvores e XGBoost com 1500. Ela considera o custo medido, sem supor superioridade de score. Execuções são sequenciais.

## Resultados medidos

| Tentativa | Escopo | AUC OOF completa | AUC validação coberta | Delta OOF | Minutos | Pico RSS GiB | Consistente |
|---|---|---:|---:|---:|---:|---:|---|
| cost_lightgbm_200 | screening [0] | — | 0.95767547 | — | 1.09 | 0.58 | não/triagem |
| cost_xgboost_200 | screening [0] | — | 0.95704050 | — | 1.64 | 0.54 | não/triagem |
| cost_catboost_200 | screening [0] | — | 0.95645545 | — | 1.42 | 0.62 | não/triagem |

## Hipóteses, inclusive resultados sem ganho

- **cost_lightgbm_200**: Cost profile: original fold 0, 200 iterations; no promotion. Deltas nos folds avaliados: [-0.00054746].
- **cost_xgboost_200**: Cost profile: original fold 0, 200 iterations; no promotion. Deltas nos folds avaliados: [-0.00118243].
- **cost_catboost_200**: Cost profile: original fold 0, 200 iterations; no promotion. Deltas nos folds avaliados: [-0.00176748].

A primeira submissão é preservada. Não há nova nota pública registrada para estes candidatos. Previsões individuais e modelos ficam fora do Git.

Fontes: [LightGBM](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [psutil](https://psutil.io/).
