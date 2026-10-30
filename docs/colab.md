# Execução local e no Google Colab

## Contextos diferentes

| Contexto | Onde o código roda | Onde os arquivos estão |
|---|---|---|
| Terminal do VS Code no Windows | Computador local | Disco local/OneDrive |
| Notebook com kernel local | Python do computador | Disco local |
| Notebook com kernel Colab | Servidor remoto | Sistema de arquivos do servidor |
| Drive montado no Colab | Acesso remoto ao Drive | Google Drive |

Salvar um arquivo no computador não o envia ao servidor. Instalar uma biblioteca no
terminal local não a instala no kernel Colab. A montagem do Drive não sincroniza a pasta
local do projeto. Arquivos em `/content` podem desaparecer quando o runtime termina.

## Extensão oficial do VS Code

Siga o [guia oficial](https://github.com/googlecolab/colab-vscode): instale a extensão
Colab, abra `notebooks/airline_satisfaction.ipynb`, faça login quando solicitado e escolha
**Select Kernel > Colab > Auto Connect**. A configuração depende de uma conta autenticada.
Não houve conexão remota ou treinamento no Colab nesta sessão.

Para execução local, selecione o Python `.venv` como kernel. O mesmo notebook chama
os módulos de `src/airline_ml`; ele não duplica o loop de treinamento.

## Código efetivamente executado

A primeira célula, no Colab, clona o repositório público em `/content`, busca `CODE_REF`
e faz checkout destacado do commit obtido. O valor inicial é `main`; o SHA resolvido
é mostrado e registrado pelos experimentos. Para reprodução, substitua `CODE_REF` pelo
SHA da execução desejada. Edições remotas pendentes bloqueiam a atualização do checkout.

O notebook aberto no VS Code é o controlador. Os módulos usados pelo kernel vêm do
checkout explícito no servidor. Ao editar módulos localmente, faça commit e push e
execute novamente a célula de atualização. Ao alterar o notebook, use o notebook da
mesma revisão do código. Não há sincronização automática.

As dependências são instaladas no Python do kernel. A base fixada foi testada em
Python 3.12.13 no Windows; a execução Linux/Colab ainda exige verificação no runtime real.
Se o pip atualizar bibliotecas já importadas, reinicie o kernel e execute novamente.

## Dados e persistência

1. Baixe os três CSVs oficiais no seu computador pelo Kaggle.
2. Envie-os ao Drive para `MyDrive/airline-s6e10/raw`, ou copie-os explicitamente para
   outra pasta do servidor e configure `DATA_DIR`.
3. Monte o Drive pelo fluxo oficial: `from google.colab import drive; drive.mount('/content/drive')`.
4. O notebook usa `MyDrive/airline-s6e10/results` para modelos, folds, logs e previsões.
5. Execute smoke, ative auditoria, revise os resultados e só então habilite piloto e treino.

Os controles `RUN_AUDIT`, `RUN_PILOT` e `RUN_FULL_TRAIN` começam desativados. O smoke
test está ativo e usa dados artificiais. O notebook não transmite credenciais ou envia
submissões ao Kaggle.

Antes de encerrar a sessão, confira `metrics.json`, `environment.json`, `config.json`,
folds, modelos e submissão no Drive. Se a sessão cair, os folds e artefatos concluídos
persistidos continuam disponíveis; o pipeline atual não retoma um treino parcialmente
concluído. Uma nova execução cria outro identificador e não sobrescreve a anterior.

Para recuperar um arquivo manualmente, use `google.colab.files.download` no servidor
ou baixe pelo Drive. Copie os resultados agregados para a documentação local após revisão.
Limpe todas as saídas antes de versionar o notebook.

## CPU, GPU e CLI

A configuração padrão usa CPU, adequada ao hardware local identificado. Uma GPU no
Colab requer selecionar um runtime compatível, confirmar `nvidia-smi` no kernel e alterar
`task_type` para `GPU` na configuração CatBoost. Não há fallback silencioso. A disponibilidade
gratuita e a duração do runtime não são garantidas. CatBoost GPU pode não ser determinístico.

A [CLI oficial Colab](https://github.com/googlecolab/google-colab-cli) declara suporte
apenas a Linux e macOS. Não foi instalada no Windows e WSL não foi avaliado; esse caminho
não é dependência do projeto. Nenhum serviço pago foi contratado ou provisionado.
