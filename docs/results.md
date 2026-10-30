# Resultados e limitações

## Competição S6E10

| Etapa | Estado |
|---|---|
| Sample inspecionado | Concluído, 299.844 registros |
| Treino e teste oficiais localizados | Concluído: 699.635 / 299.844 registros |
| Auditoria da base oficial | Concluída, sem duplicatas exatas ou IDs sobrepostos |
| Piloto com dados oficiais | Concluído: 10 mil linhas, 83,33 s |
| Baseline e OOF oficiais | Concluído: 699.635 previsões OOF, 5 folds |
| Submissão real gerada | Concluído e validado: 299.844 previsões |
| Envio ao Kaggle | Não realizado |
| Aceitação com score | Não observada |
| Execução em Colab | Não realizada |

O piloto obteve AUC OOF de 0,95347198 apenas na amostra de 10 mil linhas. A baseline
completa obteve **AUC OOF de 0,95816557**. Não existe nota pública medida no Kaggle.
As notas dos notebooks de terceiros estão em [references.md](references.md) e não entram
na comparação de experimentos próprios.

## Primeira submissão

Experimento: `20261030T184827_catboost_first_submission_06d4f0`.
Configuração: `configs/catboost-first.json`, CatBoost CPU/4 threads, depth 6,
learning rate 0,05, até 1000 árvores por fold, seed 2026. Sem dados externos.
Código executado: commit `bc5633f`, sem modificações locais no início do experimento.

| Fold (ID) | ROC AUC | Árvores preservadas | Tempo (s) |
|---|---:|---:|---:|
| 0 | 0,95822293 | 999 | 323,83 |
| 1 | 0,95806571 | 996 | 329,74 |
| 2 | 0,95844324 | 1000 | 480,98 |
| 3 | 0,95790067 | 999 | 285,21 |
| 4 | 0,95822197 | 997 | 295,83 |

AUC OOF global: **0,95816557**. Média das AUCs dos folds: 0,95817090.
Desvio padrão populacional dos folds: 0,00018087, sem interpretação como intervalo
de confiança. Duração total do pipeline: 1746,97 segundos, aproximadamente 29 min 7 s.

As probabilidades de teste são a média dos cinco modelos. Arquivo local para envio:
`submission_01_catboost.csv` (8.501.921 bytes). O conteúdo é idêntico ao `submission.csv`
arquivado na pasta do experimento. SHA-256:
`6f5a70f9854c5db2567885deaf57a087b7254edd85e3766934dfbe5534418863`.

O arquivo foi reaberto e validado pelo pipeline e pela CLI. Uma conferência independente
linha a linha confirmou cabeçalho, 299.844 registros, IDs únicos na ordem do sample,
ausência de índice extra e probabilidades finitas entre 0,00141762 e 0,99796266.
Há 299.731 probabilidades distintas. O envio e a aceitação com nota continuam pendentes.

O [resumo agregado](../reports/first_submission_summary.json) e
o [registro de experimentos](../reports/experiments.csv) são públicos. Modelos, OOF,
submissão e manifesto local em `outputs/submission_01_manifest.json` são ignorados pelo Git.
O limite de árvores foi alcançado; uma comparação futura com mais iterações ou outros
backends deve reutilizar os mesmos folds. Nenhuma escolha foi feita pelo leaderboard.

## Auditoria dos dados oficiais

O treino possui 310.339 registros positivos (`True`) e 389.296 negativos (`False`).
Há 292 nulos no atraso de chegada no treino e 130 no teste. Zeros, categorias e
outliers foram preservados. Todas as categorias do teste aparecem no treino.

![Auditoria dos dados oficiais](../reports/figures/competition-audit.png)

Relatório agregado: [competition_data_audit.json](../reports/competition_data_audit.json).
Em dez faixas consecutivas de ID, a prevalência varia de 44,18% a 44,61%.
O ID permanece excluído das features; essa verificação não substitui a investigação
de grupos ou tempo quando houver variáveis que os representem.

## Referência externa auditada

O arquivo local auditado tem 129.880 registros, 22 colunas e alvo booleano:
56.428 `True` e 73.452 `False`. Há 393 valores ausentes em `Arrival Delay in Minutes`.
Não foram encontradas linhas ou vetores de features exatamente duplicados.

Os atrasos máximos são 1.592 e 1.584, incompatíveis com a descrição de notas 0–5.
O relatório agregado e o hash do arquivo estão em
[external_reference_audit.json](../reports/external_reference_audit.json).
Esse arquivo não foi usado para treinar modelos de competição.

## Verificação funcional

Os testes usam dados artificiais, com nulos, categorias, IDs fora da ordem do sample,
duplicatas e entradas incompatíveis. São verificadas integridade dos folds, reutilização,
rejeição de adulteração do cache, cobertura OOF e leitura da submissão gravada.

Verificações executadas localmente:

- **32 testes aprovados**, incluindo treinos pequenos dos três backends. Última execução:
  15,21 segundos. Nenhum aviso nessa execução.
- Ruff: verificação estática aprovada.
- Smoke pela CLI: concluído com 240 registros de treino, 31 de teste e 3 folds.
  O arquivo gravado foi reaberto e validado. Estado registrado: `diagnostic_only`.
- Tentativas com os dados oficiais ausentes e com os CSVs incompatíveis de Downloads:
  rejeitadas com mensagens explícitas antes do treino.
- Verificação de publicação: nenhum dado individual ou segredo detectado pelos padrões
  configurados. Essa verificação não constitui garantia absoluta de ausência de segredos.

- Notebook executado em kernel local Python 3.12, incluindo instalação/configuração,
  checkout identificado e smoke test. A cópia executada fica apenas em
  `artifacts/notebook-qa/executed-local.ipynb`, ignorada pelo Git; a versão publicada
  permanece sem saídas. O caminho remoto Colab ainda não foi executado.
- Gráfico da auditoria artificial gerado e inspecionado visualmente.
- Código e documentação publicados por push confirmado no repositório público
  [EduardoTostesC/airline-satisfaction-ml](https://github.com/EduardoTostesC/airline-satisfaction-ml).
- [GitHub Actions em Ubuntu/Python 3.12](https://github.com/EduardoTostesC/airline-satisfaction-ml/actions/runs/41002072666)
  aprovado no commit `1b91244`: instalação, Ruff, testes e verificação de publicação.
  O CI instala o lock principal; os extras LightGBM e XGBoost são verificados localmente.
- [GitHub Actions da versão usada nesta etapa](https://github.com/EduardoTostesC/airline-satisfaction-ml/actions/runs/41110932912)
  também aprovado no commit `8fb9fe3`.

Scores e tempos artificiais não são benchmarks de desempenho da S6E10.
