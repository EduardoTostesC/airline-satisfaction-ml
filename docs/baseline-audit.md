# Auditoria da primeira baseline

Auditoria em 30/10/2026. Objetivo: estabelecer uma referência verificável para melhorar
a solução sem descartar experimentos. Top 10 é uma ambição, sem promessa de resultado.
Nenhum novo treinamento completo foi necessário nesta etapa.

## Identificação exata e resultado público

Arquivo preservado: `submission_01_catboost.csv`, 8.501.921 bytes e 299.844 previsões.
SHA-256 calculado novamente:

```text
6f5a70f9854c5db2567885deaf57a087b7254edd85e3766934dfbe5534418863
```

O hash é igual ao da cópia `submission.csv` do experimento
`20261030T184827_catboost_first_submission_06d4f0` e ao registrado na geração.
A nova inferência dos cinco modelos também reproduziu o CSV **byte a byte**.
Não sobrescrevemos a primeira submissão, os modelos, a OOF ou os metadados originais.

| Evidência | Valor | Origem |
|---|---:|---|
| AUC OOF recalculada | 0,958165568758 | Artefatos locais auditados |
| AUC pública | 0,95770 | Informada pelo usuário |
| Posição na captura relatada | 3020 | Informada pelo usuário |
| Nota do primeiro na captura | 0,96473 | Informada pelo usuário |
| Nota do décimo | Não fornecida | Não estimada |
| Divisão pública/privada | Aproximadamente 20% / 80% | Informada pelo usuário |

A associação entre resultado, arquivo, hash e experimento está em
[submission_01_public_result.json](../reports/submission_01_public_result.json), no
[registro de experimentos](../reports/experiments.csv) e no resumo da primeira submissão.
O estado `accepted_user_reported` identifica a procedência do relato. Não houve consulta
direta ao Kaggle nesta auditoria. Recibo/ID da submissão e horário da captura não foram
fornecidos. A posição 3020 não é apresentada como atual, privada ou definitiva.

## Inventário e cadeia de reprodução

Diretório original, referido abaixo como `RUN`:

```text
artifacts/runs/20261030T184827_catboost_first_submission_06d4f0/
```

| Artefato | Localização | Verificação |
|---|---|---|
| Código original | Git `bc5633fbac4db37d8fc45084ab23d74c1be1ca17` | Hashes dos módulos históricos conferem com o ambiente registrado |
| Configuração original | `RUN/config.json` | Snapshot preservado; a configuração atual acrescenta a trava de folds |
| Ambiente | `RUN/environment.json` | Python 3.12.13, CatBoost 1.2.10, CPU; checkout limpo no início |
| Esquema | `RUN/schema.json` | 21 features e quatro categorias verificadas |
| Cinco modelos | `RUN/model_fold_0.cbm` até `model_fold_4.cbm` | Todos os hashes conferem com `RUN/metrics.json` |
| OOF | `RUN/oof.csv` | Hash, IDs, alvos, folds, cobertura e previsões conferidos |
| Métricas | `RUN/metrics.json` e `RUN/fold_metrics.json` | AUCs reproduzidas com os modelos salvos |
| Pré-processamento | `RUN/preprocessing_fold_0.json` até `preprocessing_fold_4.json` | Categorias nativas, sem vocabulário externo aprendido |
| Log de treino | `reports/local/catboost_first.log` | UTF-16 LE; cinco resultados e melhores iterações conferem com os modelos |
| Submissão arquivada | `RUN/submission.csv` | Mesmo hash do CSV associado ao score público |

[baseline_audit.json](../reports/baseline_audit.json) contém hashes, parâmetros efetivos
por modelo e verificações. O hash do código registrado na geração é
`f549ca229d589f8af7c4a7810a6affc70aeec41a1458358fb9b86177b41a737c`.

| Dados verificados | Linhas | SHA-256 |
|---|---:|---|
| train.csv | 699635 | `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36` |
| test.csv | 299844 | `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6` |
| sample_submission.csv | 299844 | `6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71` |

## Parâmetros realmente executados

CatBoost **CPU**, quatro threads, cinco folds estratificados com shuffle. Seed **2026**
no splitter e em cada modelo. Cada fold ajustou 559.708 linhas e validou 139.927.
Não houve refit completo: a submissão é a média aritmética das cinco probabilidades
de classe 1, com peso 1/5 para cada modelo.

| Parâmetro | Valor recuperado |
|---|---|
| iterations / depth / max_leaves | 1000 / 6 / 64 |
| learning_rate | 0,05; valor interno aproximado 0,05000000075 |
| loss_function / eval_metric | Logloss / AUC |
| l2_leaf_reg / border_count | 3 / 128 |
| nan_mode / task_type / thread_count | Min / CPU / 4 |
| random_seed / od_type / od_wait | 2026 / Iter / 100 |
| use_best_model | true |
| boosting_type / grow_policy | Plain / SymmetricTree |
| bootstrap_type / subsample | MVS / aproximadamente 0,8 |
| rsm / random_strength | 1 / 1 |
| one_hot_max_size / max_ctr_complexity | 2 / 4 |
| counter_calc_method | SkipTest |
| leaf_estimation_method / iterations | Newton / 10 |
| auto_class_weights / has_time | None / false |

Os defaults foram recuperados via `get_all_params`, que inclui parâmetros calculados
no treino, conforme a [documentação oficial](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_all_params).
Os parâmetros completos e as opções nativas de execução por fold estão no JSON da auditoria.

| Fold | Melhor índice, base zero | Árvores preservadas | Iterações avaliadas | AUC reproduzida |
|---|---:|---:|---:|---:|
| 0 | 998 | 999 | 1000 | 0,958222930859 |
| 1 | 995 | 996 | 1000 | 0,958065705196 |
| 2 | 999 | 1000 | 1000 | 0,958443236435 |
| 3 | 998 | 999 | 1000 | 0,957900674394 |
| 4 | 996 | 997 | 1000 | 0,958221971685 |

Early stopping estava configurado, mas **não interrompeu o treino antes do limite**
de 1000 iterações. O modelo salvo foi reduzido ao melhor prefixo pela AUC de validação.
O campo histórico `best_iterations` representa quantidade de árvores, não índice base
zero. Não foi encontrado erro de uma posição na seleção.

## Features, categorias, nulos e fontes adicionais

Entraram, nesta ordem, as 21 features:

```text
Age, Flight Distance, Inflight wifi service, Departure/Arrival time convenient,
Ease of Online booking, Gate location, Food and drink, Online boarding,
Seat comfort, Inflight entertainment, On-board service, Leg room service,
Baggage handling, Checkin service, Cleanliness, Departure Delay in Minutes,
Arrival Delay in Minutes, Gender, Customer Type, Type of Travel, Class
```

Os nomes internos são `f0` a `f20`. As quatro categorias são `Gender`, `Customer Type`,
`Type of Travel` e `Class`, nos índices 17–20. Nomes, ordem e índices conferem nos cinco
modelos. Nenhuma feature foi descartada ou listada em `ignored_features`. Todas têm
importância não nula em cada modelo; isso não demonstra causalidade.

Categorias recebem conversão determinística: `V:` antes do valor observado e `M:`
para nulo. Não houve one-hot ou target encoding externo. O CatBoost ajustou seu
tratamento nativo dentro de cada treino, com `counter_calc_method=SkipTest`.
Não foi encontrado transformador aprendido fora do fold.

Numéricas foram convertidas para float64, preservando zeros, outliers e NaNs. O
CatBoost usou `nan_mode=Min`. Existem 292 nulos no atraso de chegada no treino e 130
no teste. Não houve imputação aprendida com validação/teste, normalização, SMOTE,
balanceamento, engenharia manual, pseudo-labeling ou carregamento do `data.csv` externo.

`id` e `satisfaction` foram excluídos das features. O alvo foi conferido linha a linha:
`False → 0`, `True → 1`. Os modelos têm classes `[0,1]` e usaram `predict_proba[:,1]`.
Inverter as probabilidades produziria AUC OOF aproximada de 0,04183443, incompatível
com as previsões reproduzidas.

## OOF, exclusão do treino e limites da evidência

Verificações executadas:

- 699.635 IDs únicos, sem faltas ou extras; alvo e fold conferidos por ID com o treino.
- Ordem OOF igual à do treino; o verificador também alinha por ID se a ordem mudar.
- Uma e somente uma previsão OOF por linha, do modelo do fold designado.
- A associação persistida reproduz `StratifiedKFold(5, shuffle=True, random_state=2026)`.
- O código histórico usa `fit_index = folds != fold` e `valid_index = folds == fold`.
  Os conjuntos reconstruídos são disjuntos em todos os folds.
- Todas as previsões OOF foram recalculadas com os CBMs: **erro máximo zero**.
- AUC global reproduzida; a média dos cinco modelos reproduz o CSV byte a byte.

As evidências sustentam a associação OOF/modelo/fold e não apontam uso da linha avaliada
como amostra de ajuste das árvores. Duas limitações permanecem explícitas:

1. Os CBMs não contêm os IDs de treinamento e não foi salvo manifesto contemporâneo
   desses IDs em cada `fit`. A reconstrução usa código histórico, folds e reprodução
   de previsões; não é uma prova independente da composição histórica do treino.
   Não criamos manifestos retrospectivos como se fossem originais.
2. O rótulo do fold avaliado participa da AUC usada para selecionar a melhor iteração.
   Isso pode introduzir viés de seleção na OOF. A linha fica fora do ajuste das árvores,
   mas não é um holdout completamente intocado pela seleção do modelo.

## Folds congelados e correções executadas

Associação original preservada:

```text
artifacts/folds/36877817064bfa9e79bd6cbbc7e30f9b79d508a9cba592fb73a7ae237ad8b962/folds.csv
```

SHA-256 do CSV: `f8f4c46125130a2ab9166c6d47bcee84061d2df67064f34cdddcef1dfd996702`.
O [manifesto de trava](../configs/folds-lock.json) fixa hash do treino, seed, estratégia,
quantidade de folds, alvo e hash do arquivo ID/fold. Os IDs individuais ficam fora do Git.

As configurações CatBoost, LightGBM e XGBoost agora apontam para a mesma trava. Ausência
ou adulteração do arquivo, dados/seed/estratégia diferentes são rejeitados em vez de
gerar outra divisão silenciosamente. Trocar backend ou features mantém a associação.
Não houve alteração da divisão nem necessidade de recalcular a baseline.

No Colab/outro computador, copie a pasta original com `folds.csv` e `metadata.json`
para `output_dir/folds/<chave>/`. O manifesto sozinho não contém os IDs. Pilotos usam
amostras/folds próprios e não substituem os folds congelados da base completa.

Correções e reforços desta etapa:

- Atualizados nota pública e estados pendentes, associados ao hash e ao experimento,
  com procedência do relato registrada.
- Adicionada a trava de folds com testes de seed, adulteração, metadados e arquivo ausente.
- Novos treinos passam a salvar contagens e hashes dos IDs selecionados para treino e
  validação no arquivo de pré-processamento de cada fold, antes do `fit`. Os artefatos
  antigos não receberam informação retrospectiva apresentada como contemporânea.
- Adicionado verificador repetível de OOF e reprodução, com testes de alinhamento,
  cobertura, IDs, alvos, folds e probabilidades corrompidos.

Nenhum defeito numérico da primeira submissão foi identificado. Nenhuma probabilidade
foi corrigida ou substituída. Metadados originais de geração permanecem imutáveis;
o resultado público posterior está em registro separado.

## Comparação dos scores e hipóteses priorizadas

OOF menos pública: **0,000465568758**. A OOF usa o treino em validação cruzada; a pública
usa uma parcela diferente do teste, aproximadamente 20% segundo o relato. Uma única
comparação não comprova a qualidade da validação e não permite atribuir o desvio a
overfitting, azar, distribuição, CPU/GPU ou um bug específico.

O desvio padrão dos folds, 0,00018087, não é IC nem tolerância para comparar OOF com
leaderboard. Os outros aproximadamente 80% compõem a avaliação privada relatada;
não há score privado conhecido. A diferença para o primeiro na captura é 0,00703.
Isso não é a diferença para top 10: falta a nota do décimo. Não estimamos posição final.

Próximas hipóteses, sem resultados presumidos e sem treinos novos nesta auditoria:

| Prioridade | Experimento proposto | Evidência e critério |
|---|---|---|
| 1 | CatBoost com até 2000 árvores, demais parâmetros e folds iguais | Todos chegaram ao cap; AUC de 500 árvores até o melhor prefixo aumentou em 0,00043–0,00059. Justifica testar, sem provar ganho futuro ou causa do score público. |
| 2 | Uma execução LightGBM e depois XGBoost com configurações existentes | Comparar OOF e duração nos mesmos folds; não há resultado próprio oficial desses modelos ainda. |
| 3 | Poucas features justificadas, uma alteração por comparação | Agregados de notas e indicadores de zero/nulo, preservando zeros e sem interpretá-los automaticamente como ausência. |
| 4 | Média/ensemble simples com OOF alinhadas | Somente após medir diversidade e ganho OOF; não ajustar pesos apenas pelo placar público. |
| Condicional | Dados externos ou pseudo-labeling | Exigem verificar regras, origem e desenho sem contaminação; não foram usados na baseline. |

Orçamento inicial proposto: uma execução completa por comparação, sem Optuna nesta etapa.
Mais tentativas exigem orçamento explícito. Só mudar a validação diante de evidência
concreta de grupos, tempo ou contaminação; nesse caso, repetir a baseline.

## Repetir a auditoria

```powershell
.venv/Scripts/python.exe scripts/audit_baseline.py --run-dir artifacts/runs/20261030T184827_catboost_first_submission_06d4f0 --submission submission_01_catboost.csv --output-dir artifacts/audits/submission_01_replay_new
```

A saída precisa ser uma pasta nova. O script usa lotes, confere o código histórico
e exige os módulos originais de pré-processamento/inferência. Se esses módulos mudarem,
use um checkout apropriado com o verificador disponível. As previsões reproduzidas ficam
somente em `artifacts/`, sem sobrescrever as anteriores.
