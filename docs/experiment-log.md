# Registro de experimentos e decisões

## 30/10/2026 — preparação

Hipótese: uma baseline tabular simples e uma validação reutilizável devem anteceder
engenharia de features, uso de fontes externas e ensembles.

Implementado: leitura e rejeição de esquemas incompatíveis, mapeamento explícito de
alvo textual, auditoria, folds persistidos, agrupamento de duplicatas, CatBoost,
LightGBM, XGBoost, early stopping, OOF, submissão validada após gravação,
registro de proveniência e notebook para execução local/Colab.

Executado: criação do `.venv` com Python 3.12.13, instalação de dependências,
auditoria da referência externa e testes pequenos de integração em CPU.
As execuções artificiais exercitam os backends, não medem qualidade na competição.

Primeira verificação: 27 testes passaram e 2 foram omitidos porque os extras de
boosting ainda não estavam instalados. Depois da instalação, 29 passaram, incluindo
os três backends. Avisos de depreciação do LightGBM motivaram atualização para os
argumentos documentados `eval_X` e `eval_y`; os checks finais são registrados em
[results.md](results.md).

Verificação final local: 32 testes aprovados sem avisos, Ruff aprovado, smoke pela CLI
concluído, notebook executado em kernel local e gráfico da auditoria artificial
inspecionado. O primeiro commit foi publicado por push confirmado. Não houve execução
no Colab, treinamento oficial ou envio ao Kaggle.

O CI Ubuntu/Python 3.12 também concluiu com sucesso no commit `1b91244`, incluindo
instalação do lock principal, Ruff, pytest e a verificação dos arquivos públicos.
O link da execução está em `results.md`; os dois modelos opcionais foram testados localmente.

Bloqueio observado: `data/raw/train.csv` ausente. A tentativa de usar Downloads
é rejeitada por ausência do ID esperado, pois os arquivos encontrados pertencem
a outro problema. Isso não é uma falha de treinamento nem resultado de competição.

Decisão: manter o pipeline preparado e aguardar os CSVs oficiais. Não substituir o
treino sintético pelo dataset externo e não registrar as notas dos notebooks públicos
como resultados próprios. Nenhum tuning foi iniciado e nenhuma previsão real foi enviada.

## 30/10/2026 — recebimento dos CSVs oficiais e piloto

Os CSVs adicionados à raiz correspondem ao sample: 699.635 linhas de treino,
299.844 de teste e 21 features. Foram preservados os originais e feitas cópias em
`data/raw`. A auditoria não encontrou duplicatas exatas ou sobreposição de IDs/features.
As quatro features categóricas não apresentam categorias novas no teste. Foram
preservados 292 nulos de atraso de chegada no treino e 130 no teste.

Piloto `20261030T184506_catboost_baseline_pilot_599c72`: 10 mil linhas estratificadas,
5 folds, CPU/4 threads, máximo de 2000 árvores, early stopping de 100. Concluiu em
83,33 segundos com AUC OOF 0,95347198. Esse resultado pertence apenas ao piloto e
seu arquivo de diagnóstico não é a primeira submissão.

Decisão: treinar `configs/catboost-first.json` com todas as linhas, 5 folds e orçamento
de 1000 árvores por fold, preservando seed e método. Logs passam a mostrar progresso
a cada 100 iterações. A duração do piloto não foi extrapolada linearmente como previsão
precisa de duração do treino completo.

## Registro automático

`reports/experiments.csv` guarda agregados de execuções completas. Pilotos e smoke tests usam registros
separados, ignorados pelo Git. Configuração completa, hashes, métricas por fold,
ambiente e artefatos ficam na pasta de cada execução local.
