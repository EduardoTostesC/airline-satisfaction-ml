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

## 30/10/2026 — primeira baseline completa e arquivo para envio

Experimento `20261030T184827_catboost_first_submission_06d4f0`, código `bc5633f`
sem alterações locais no início. Treino completo em CPU/4 threads, cinco folds
estratificados e 1000 árvores máximas por fold. Todas as 699.635 linhas receberam
uma previsão OOF; as probabilidades do teste resultam da média dos cinco modelos.

Resultado: AUC OOF 0,95816557; média dos folds 0,95817090; desvio padrão populacional
0,00018087. Tempo total: 1746,97 s. Métricas detalhadas e tempos por fold estão em
`results.md` e `reports/first_submission_summary.json`.

Foi criada uma cópia local `submission_01_catboost.csv` para o usuário submeter.
Validações do pipeline, CLI e conferência independente confirmaram 299.844 IDs únicos,
ordem do sample, cabeçalho exato e probabilidades finitas no intervalo [0,1].
O SHA-256 confere com o arquivo arquivado. Estado: **gerado e validado**, sem envio
ao Kaggle ou score observado. Dados, modelos, OOF e submissão permanecem fora do Git.

Decisão: entregar esta baseline como primeira submissão reproduzível. O orçamento de
árvores foi atingido; aumentá-lo e comparar outros backends são experimentos posteriores,
com os mesmos folds e orçamento medido. O dataset externo não foi usado.

## 30/10/2026 — resultado público e auditoria da baseline

O usuário informou AUC pública **0,95770** para `submission_01_catboost.csv`, posição
3020 e score do primeiro 0,96473 na captura relatada. Não informou nota do décimo,
horário da captura, ID de submissão ou score privado. A divisão aproximada 20%/80%
também veio do relato. A associação está em `reports/submission_01_public_result.json`.

O hash do CSV foi recalculado e confere com o experimento original. Foram localizados
os cinco modelos, OOF, folds, configuração, ambiente e log de treino. A inferência
reproduziu **todas as 699.635 previsões OOF com erro zero** e o CSV de teste byte a byte.
Não houve novo treino completo nem alteração da primeira submissão.

Confirmados CPU/4 threads, seed 2026, 5 folds, 21 features sem ID/alvo, quatro categorias,
nulos numéricos nativos, classe positiva True/1 e média dos folds sem refit. Parâmetros
efetivos foram recuperados dos modelos. Os cinco avaliaram 1000 iterações: o early
stopping não interrompeu antes do cap. Não foi identificado defeito numérico que
justificasse corrigir as probabilidades.

OOF 0,95816557 menos pública 0,95770 = 0,00046557, em conjuntos diferentes. Essa única
comparação não comprova a validação nem identifica causa para o score. A ausência de
manifestos contemporâneos dos IDs de treino nos artefatos originais é uma limitação,
não foi preenchida com evidência retrospectiva. O rótulo da validação participa da
seleção da melhor iteração; esse viés potencial está explicitado na auditoria.

Correções: estados/score atualizados com procedência; trava de folds nos próximos
experimentos; hashes dos IDs selecionados em novos treinos; testes de OOF e integridade
da trava. Os dados e os folds originais foram preservados. **44 testes aprovados**.

Prioridades: testar orçamento de 2000 árvores, comparar LightGBM/XGBoost nos mesmos
folds e só depois poucas features/ensemble medidos por OOF. Top 10 é ambição, não
previsão. Não foi iniciada busca nem experimento completo novo nesta auditoria.
Diagnóstico detalhado: [baseline-audit.md](baseline-audit.md).

## Registro automático

`reports/experiments.csv` guarda agregados de execuções completas. Pilotos e smoke tests usam registros
separados, ignorados pelo Git. Configuração completa, hashes, métricas por fold,
ambiente e artefatos ficam na pasta de cada execução local.
