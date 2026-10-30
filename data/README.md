# Dados

Fonte obrigatória: [Kaggle Playground S6E10](https://www.kaggle.com/competitions/playground-series-s6e10/data).
Entre na conta e revise/aceite as regras pelo fluxo oficial. Baixe os três arquivos:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Também é possível usar outro diretório com `--data-dir` ou `data_dir` na configuração.
Os três arquivos precisam estar na mesma pasta. Dados e arquivos ZIP são ignorados pelo Git.

## Obtenção pela CLI oficial

Opcionalmente, instale o extra `.[kaggle]` no ambiente isolado e use:

```powershell
.venv/Scripts/kaggle.exe auth login
.venv/Scripts/kaggle.exe competitions download playground-series-s6e10 -p data/raw
```

Extraia o ZIP nessa pasta. A autenticação usa o navegador oficial no computador.
Não coloque credenciais no código, notebook, Git ou chat.
Consulte a [documentação de autenticação](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).

## O que foi observado em 30/10/2026

- Sample recebido: 299.844 registros, IDs únicos 699635–999478, colunas `id,satisfaction`.
- SHA-256 do sample: `6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71`.
- Todas as probabilidades do sample: `0.44357272006117476`. Isso não comprova a prevalência do alvo.
- Os CSVs adicionados depois à raiz foram validados e copiados para `data/raw`:
  699.635 linhas de treino, 299.844 de teste e 21 features (17 numéricas e 4 categóricas).
  O alvo é booleano: `True` corresponde à classe positiva (310.339 registros) e
  `False` à negativa (389.296 registros). A prevalência observada é 44,3573%.
  Há 292 nulos no atraso de chegada no treino e 130 no teste. Não foram identificadas
  duplicatas exatas de features nem sobreposição exata de features ou IDs entre conjuntos.
  O relatório agregado está em `reports/competition_data_audit.json`.
- Os arquivos `train.csv` e `test.csv` encontrados em Downloads usam `item_id`,
  têm 8.964 e 669 registros e são incompatíveis com a S6E10.
- Um `data.csv` externo também foi localizado em Downloads. Sua auditoria é separada,
  sem substituir o treino da competição e sem usar esse dado no pipeline por padrão.

## Dataset externo

O material fornecido descreve **Airline Satisfaction Data**, de Arseniy Shutko,
com 129.880 pesquisas e 22 colunas. A descrição declara CC0, mas o endereço específico
e a cadeia de origem/licenciamento ainda precisam ser confirmados. Não redistribuímos o arquivo.
O fato de notebooks públicos usarem uma fonte externa não confirma sua admissibilidade
pelas regras atuais da competição.

As afirmações de coleta real, anonimização e licença são declarações do publicador,
não verificações independentes deste projeto. Nomes como `Arrival Delay in Minutes`
exigem conferir os valores: não assumimos que esses atrasos sejam notas de 0 a 5.
