# Metodologia

## Problema e escopo

Classificação binária com probabilidades da classe 1 para `satisfaction`, avaliada por
ROC AUC. A interpretação de 1 deve ser confirmada no treino oficial. Rótulos textuais
exigem `target_mapping` explícito, por exemplo `{"satisfied": 1, "neutral or dissatisfied": 0}`
somente se esses forem os rótulos realmente observados.

O baseline usa apenas o treino oficial. Não há SMOTE, normalização, balanceamento,
pseudo-labeling, dados externos ou seleção de features automática. Zeros e outliers
são preservados. O ID nunca entra nas features.

## Auditoria

`airline-ml audit` verifica cabeçalhos, colunas, alvo, tipos, nulos, unicidade e
correspondência de IDs. Produz quantis numéricos, contagem de zeros, categorias novas
no teste e distância de variação total das distribuições categóricas.

As duplicatas exatas são identificadas por hash de todas as features originais, sem ID
ou alvo, antes da seleção de features. Há uma possibilidade teórica de colisão de hash.
O relatório informa duplicatas, grupos com alvos conflitantes e sobreposição exata de
features entre treino e teste. Sobreposição de IDs é rejeitada.

Tipos numéricos não distinguem automaticamente escalas ordinais de grandezas contínuas.
A identificação de passageiros repetidos, grupos e estrutura temporal exige revisão
dos dados e do domínio. O pipeline não oferece validação temporal genérica: uma necessidade
temporal exige implementar e documentar uma divisão apropriada antes do treino completo.

## Folds

Configuração inicial: 5 folds, shuffle, seed 2026. A auditoria do treino oficial
não identificou duplicatas exatas, coluna de tempo absoluto ou identificador de grupo
além do ID da amostra. Para a primeira submissão, foi mantido `StratifiedKFold`.
As variáveis de horário conveniente e atraso são características da pesquisa, não
uma cronologia dos registros. Sem grupos ou duplicatas, usa `StratifiedKFold`.
Com duplicatas exatas ou `group_column`, usa `StratifiedGroupKFold`. Uma coluna de grupo
configurada é excluída das features. A estratificação por grupos pode ser aproximada;
cada treino e validação precisa conter as duas classes ou o pipeline rejeita a divisão.

A tabela `id,fold` fica em `artifacts/folds/<chave>/`, acompanhada por hash do CSV,
hash do treino, estratégia, seed e configuração do alvo. O cache é verificado antes de
reutilização. Mudar de modelo ou retirar uma feature preserva a divisão; mudar dados,
seed, alvo ou estratégia pode exigir uma nova divisão e repetir a baseline.

Após auditar a primeira submissão, `configs/folds-lock.json` passou a fixar os folds
originais nas configurações completas. Alterações de dados/seed/estratégia, CSV
adulterado ou cache ausente interrompem a execução em vez de criar outra divisão.
Outro runtime deve receber a pasta original `folds/<chave>/`. Pilotos continuam com
folds próprios, sem substituir a referência. Nenhum fold da baseline foi alterado.

## Modelos e pré-processamento

CatBoost recebe categorias como strings, com representações distintas para nulo e valor
observado. Categorias numéricas inteiras usam representação canônica consistente entre
treino e teste. Nulos numéricos permanecem nulos. Categorias novas são processadas nativamente.

LightGBM e XGBoost aprendem o vocabulário categórico somente no treino de cada fold.
Categorias não vistas nesse treino são tratadas como ausentes na validação e no teste.
O mapeamento de cada fold é salvo. Não há target encoding externo.

Todos os modelos usam early stopping por AUC. O fold de validação também escolhe a melhor
iteração, portanto a métrica tem o viés de seleção associado a essa escolha. Ajustes
extensos exigiriam validação adicional; os resultados não são estimativas sem viés de
generalização para o mundo real. O treino do teste não recebe rótulos nem entra no ajuste
do vocabulário das categorias.

As previsões de teste são a média simples das probabilidades dos modelos dos folds.
O pipeline não faz um refit adicional em todo o treino. Cada modelo, seu esquema e os
metadados necessários ficam na pasta da execução.

A configuração `configs/catboost-first.json` limita a primeira submissão a 1000 árvores
por fold, com early stopping de 100 iterações e quatro threads em CPU. A configuração
de 2000 árvores permanece disponível para um experimento posterior nos mesmos folds.
O treino usa todas as linhas oficiais, sem incorporar o dataset externo.

## Métricas e rastreabilidade

A OOF precisa cobrir cada registro exatamente uma vez. São registrados AUC global,
AUC por fold, média e desvio padrão populacional dos folds. Esse desvio **não é intervalo
de confiança**. Importâncias internas dos modelos não são explicações causais e as
escalas dos diferentes backends não devem ser comparadas diretamente.

Cada execução registra configuração, hipótese, hashes dos três CSVs, hash do código,
commit, estado com edições locais, versões, hardware observado, dispositivo solicitado,
tempo e número de iterações por fold. Modelos e previsões recebem hashes. Falhas de treino
geram `failure.json` e não são registradas como experimentos concluídos.

Novas execuções registram também hashes e contagens dos IDs selecionados para treino
e validação em `preprocessing_fold_<id>.json`. Esses manifestos não existiam na primeira
execução; sua ausência histórica e os limites da inferência sobre composição do treino
estão em [baseline-audit.md](baseline-audit.md). Os modelos originais permitiram recuperar
parâmetros efetivos e reproduzir todas as previsões sem treinar novamente.

O registro CSV agregado é atualizado ao finalizar uma execução. Use um processo de treino
por registro; não há coordenação de gravações concorrentes.

## Piloto e testes

`--pilot-rows` seleciona uma amostra estratificada reproduzível do treino, cria chave de
fold própria e escreve em `artifacts/pilots/`. Produz `diagnostic_predictions.csv`, sem
submissão final. O smoke test usa dados artificiais, 3 folds e 30 iterações máximas.

Tempos artificiais não estimam o treino completo. A estimativa inicial depende de um
piloto com os CSVs oficiais no dispositivo que será efetivamente usado. A relação entre
linhas, complexidade categórica, early stopping e duração não é necessariamente linear.

## Submissão

O alinhamento usa a chave ID e a ordem do sample, mesmo que o teste esteja em ordem
diferente. O arquivo é reaberto e validado após a gravação: cabeçalho, número de linhas,
IDs, ordem, ausência de índice extra e probabilidades finitas no intervalo [0,1].
Não há arredondamento para classes. O envio ao Kaggle é uma etapa externa, com evidência
separada de aceitação e score.
