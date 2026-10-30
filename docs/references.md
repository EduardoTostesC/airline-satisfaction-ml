# Fontes, notebooks de referência e regras

Consulta e análise: **30/10/2026**. Metadados fornecidos pelo usuário podem refletir
uma versão diferente da indexada na web. Nenhum código desses notebooks foi copiado.

## Competição

[Predicting Airline Satisfaction — Playground S6E10](https://www.kaggle.com/competitions/playground-series-s6e10).
A descrição fornecida informa dados sintéticos, alvo `satisfaction`, ROC AUC e submissão
`id,satisfaction`. Informa prazo de 31/10/2026 às 23:59 UTC, equivalente a 20:59 de Brasília.
A página consultada não retornou conteúdo legível suficiente para reconfirmar o prazo.

[Regras específicas](https://www.kaggle.com/competitions/playground-series-s6e10/rules):
inacessíveis pela ferramenta de consulta nesta sessão.

| Item | Estado |
|---|---|
| Prazo atualizado | Não verificado independentemente |
| Limite diário de submissões | Não verificado |
| Quantidade/seleção dos arquivos finais | Não verificado |
| Regras de equipes | Não verificadas |
| Assistência por IA | Não verificada |
| Dados externos | Não verificados |
| Compartilhamento de código e dados | Não verificado |

O uso de uma fonte por outro participante não confirma permissão. Não substituímos
essas lacunas por regras de outra edição. Dados e previsões individuais ficam fora do Git.

## Notebooks indicados pelo usuário

| Referência | Versão fornecida | AUC pública fornecida | Melhor pública informada | Tempo informado |
|---|---|---|---|---|
| Anhad Mahajan — S6E10, Are You Satisfied | V33 | 0.96088 | 0.96110, V30 | 1h12m29s |
| Дворкин Евгений Владимирович — S6E10 LightGBM CV 5 folds 0.96111 | V14 | 0.96069 | 0.96069, V14 | 19m54s |

Ambos mostram GPU T4 x2 e licença Apache 2.0 nos metadados fornecidos. A identificação
do hardware do runtime não comprova utilização das duas GPUs pelo treinamento.
Esses tempos não estimam execução local em CPU.

[Notebook de Anhad Mahajan](https://www.kaggle.com/code/anhadmahajan06/s6e10-are-you-satisfied):
a busca retornou uma versão indexada V31, diferente da V33 fornecida. O índice indica
engenharia de features, vários modelos, pseudo-labeling e ensembles. O conteúdo completo
de código não ficou acessível nesta consulta; não foi possível auditar a implementação
ou reproduzir o método. Não atribuímos sua AUC ao nosso pipeline.

O endereço exato do segundo notebook não foi localizado com segurança. Seu título
contém `CV 5 folds 0.96111`; esse número não é o score público `0.96069` fornecido.
Sem código, dados, folds e OOF, não é possível verificar a validação.

Decisão: usar a comparação como motivação para uma baseline simples e medições próprias.
Dados externos, pseudo-labeling, otimização e ensembles ficam posteriores à baseline
oficial e à verificação das regras. Nenhum score de referência é uma promessa de resultado.

## Dataset externo fornecido

O anexo descreve **Airline Satisfaction Data**, de Arseniy Shutko, com 129.880 pesquisas,
22 colunas, alvo booleano e declaração CC0. O endereço exato e a origem dos direitos não
foram confirmados. A auditoria do arquivo local está em
[external_reference_audit.json](../reports/external_reference_audit.json).

A descrição diz que atrasos são notas de 0 a 5, mas os valores observados chegam a
1592 e 1584, respectivamente. Preservamos os valores e interpretamos essa discrepância
como erro da descrição, não como autorização para transformar os dados.

## Documentação técnica primária

- [CatBoost: categorias](https://catboost.ai/docs/en/features/categorical-features)
- [CatBoost: valores ausentes](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing)
- [CatBoost: FAQ e diferenças CPU/GPU](https://catboost.ai/docs/en/concepts/faq)
- [LightGBM: LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [XGBoost: categorias e serialização](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
- [Kaggle CLI: autenticação](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Colab no VS Code](https://github.com/googlecolab/colab-vscode)
- [Colab CLI: plataformas suportadas](https://github.com/googlecolab/google-colab-cli)

A implementação foi escrita com assistência do Codex a partir dos requisitos do projeto
e das APIs documentadas. Não foram reutilizadas células dos notebooks citados.
As dependências mantêm suas próprias licenças. Não foi definida uma licença de redistribuição
para os dados nem uma licença aberta para o código do projeto nesta etapa.
