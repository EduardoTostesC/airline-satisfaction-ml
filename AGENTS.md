# Manutenção do projeto

- Autor: Adilson Eduardo Tostes Carmo. Documentação principal em português.
- Use o ambiente `.venv` e mantenha o código principal em `src/airline_ml`.
- Leia README e docs antes de mudar a metodologia. Atualize a documentação junto da mudança.
- Nunca publique dados, modelos, credenciais, previsões individuais, OOF ou submissões.
  Revise também as saídas dos notebooks. `reports/experiments.csv` contém apenas agregados.
- Não confunda o dataset externo de 129.880 pesquisas com o treino sintético da competição.
- Não registre resultados de terceiros como resultados reproduzidos neste projeto.
- Não invente features, dimensão do treino, score, ranking, tempo de execução ou estado de envio.
- Preserve zeros, categorias e outliers até haver hipótese documentada e comparação medida.
- Exclua ID das features. Ajuste qualquer transformação aprendida somente no treino do fold.
- Reutilize folds associados ao hash do treino. Mudanças da validação exigem repetir a baseline.
- Investigue duplicatas, grupos e tempo antes do treino completo. Não divida grupos entre folds.
- Probabilidades da classe 1 e ROC AUC são obrigatórias. Desvio dos folds não é IC.
- Execute `python -m pytest -q`, `ruff check src tests scripts` e
  `python scripts/check_publication.py` antes de commits relevantes.
- CI usa somente dados artificiais e treinos pequenos; nunca dados da competição ou credenciais.
- Execute um piloto antes de recomendar treino completo. Orçamentos de tuning devem ser explícitos.
- CatBoost é a primeira baseline. Comparações posteriores usam os mesmos folds.
- Autenticações ocorrem pelo fluxo oficial local. Não solicite tokens no chat.
- Não contrate recursos pagos ou tente contornar limites de execução remota.
- Código local, kernel remoto, arquivos do servidor e Drive são contextos distintos.
- O notebook deve registrar o commit e o hash do código realmente executado.
- Não force push, não reescreva histórico e confirme o remote antes de publicar.
- Diferencie arquivo gerado, enviado e aceito com nota. Registre evidência de cada etapa.
