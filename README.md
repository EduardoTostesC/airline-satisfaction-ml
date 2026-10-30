# Predicting Airline Satisfaction

Projeto de machine learning para o Kaggle Playground Series S6E10.

Autor: **Adilson Eduardo Tostes Carmo**.

## Status

Pipeline implementado e verificado com **32 testes aprovados**, incluindo os três
backends em dados artificiais. Dados oficiais recebidos: 699.635 linhas de treino,
299.844 de teste e 21 features. Auditoria e piloto concluídos; o primeiro treinamento
completo CatBoost está em execução em CPU. O notebook também executou com sucesso
em kernel local. Ainda não há nota pública do Kaggle para este projeto. Resultados
de notebooks de terceiros não são resultados reproduzidos aqui.

## Ambiente e instalação

Python 3.12 é a versão usada no diagnóstico local. Instale o projeto em ambiente isolado:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe -m pytest -q
```

Sem uv: use `py -3.12 -m venv .venv` e o pip desse ambiente. Em Linux/Colab,
substitua `.venv/Scripts/python.exe` por `.venv/bin/python` quando usar ambiente local.

## Execução

Coloque os três CSVs oficiais em `data/raw/`, conforme [data/README.md](data/README.md).

```powershell
.venv/Scripts/airline-ml.exe smoke
.venv/Scripts/airline-ml.exe audit --config configs/catboost.json
.venv/Scripts/airline-ml.exe train --config configs/catboost.json --pilot-rows 10000
.venv/Scripts/airline-ml.exe train --config configs/catboost-first.json
```

O piloto usa amostragem estratificada do treino e escreve em uma área separada.
Ele serve para verificar funcionamento e medir custo, não para comparação com CV completa.
O primeiro treino completo limita cada um dos cinco folds a 1000 árvores; a configuração
`catboost.json` mantém um orçamento de 2000 para comparação posterior.
Cada treino completo salva modelos por fold, OOF, metadados e `submission.csv` em
`artifacts/runs/<identificador>/`. A submissão usa a média dos modelos dos folds.

Revise `audit.json` antes do treino completo, especialmente grupos, duplicatas e possível
estrutura temporal. O pipeline não infere o significado de todas as features.

## Método e resultados

A métrica principal é ROC AUC da classe 1. `id` fica fora das features. A validação
estratificada usa grupos quando há duplicatas exatas ou `group_column` configurada.
CatBoost é a primeira baseline. LightGBM e XGBoost possuem configurações para comparação
nos mesmos folds. Para instalar também as versões fixadas desses dois modelos:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-boosting-lock.txt
```

- [Metodologia](docs/methodology.md)
- [Resultados e limitações](docs/results.md)
- [Registro de experimentos](docs/experiment-log.md)
- [Execução no Colab](docs/colab.md)
- [Referências e regras](docs/references.md)
- [Diagnóstico do ambiente](docs/environment.md)

## Submissão e publicação

Geração local, envio ao Kaggle e aceitação com nota são etapas distintas. O pipeline
apenas gera e verifica arquivos. Depois de um treino completo, envie o `submission.csv`
da execução escolhida pelo botão **Submit Prediction** na página oficial e registre o
resultado retornado pelo Kaggle com o identificador do experimento.

Os dados, modelos, previsões e credenciais ficam fora do Git. Apenas código, documentação,
configurações e resultados agregados são publicados. Não há licença de dados de terceiros
concedida por este repositório. Consulte as condições de cada fonte.

O dataset da competição é sintético, conforme a descrição fornecida. Este projeto não é
um sistema validado para decisões reais de companhias aéreas. A implementação recebeu
assistência do Codex; fontes consultadas e limites da reprodução estão documentados.
