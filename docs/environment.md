# Ambiente observado

Diagnóstico em **30/10/2026**, no computador do projeto.

| Componente | Resultado |
|---|---|
| Sistema | Windows 11 Pro |
| Shell | Windows PowerShell 5.1.26100.9711 |
| CPU | Intel Core i5-13420H, 8 núcleos / 12 processadores lógicos |
| RAM utilizável | 7,73 GiB |
| RAM livre na primeira consulta | Aproximadamente 0,70 GiB; medida variável |
| Vídeo detectado | Intel UHD Graphics |
| NVIDIA | `nvidia-smi` indisponível |
| Python disponíveis | 3.13.14, 3.12.13 e 3.11.15 |
| Python selecionado | 3.12.13, `.venv` isolado |
| Git | 2.55.0.windows.3 |
| uv | 0.11.15 |
| GitHub CLI | Ausente no PATH |

O conector GitHub autenticado confirmou o repositório público
`EduardoTostesC/airline-satisfaction-ml` e permissão de escrita. Inicialmente, o
repositório remoto estava vazio e o checkout local não tinha commits.

O Python global não foi alterado. Dependências foram instaladas no `.venv`:
CatBoost 1.2.10, pandas 2.3.3, NumPy 2.5.4, scikit-learn 1.9.1,
LightGBM 4.7.0 e XGBoost 3.4.1. Os arquivos `requirements-lock.txt` e
`requirements-boosting-lock.txt` fixam as resoluções usadas.

O lock principal inclui ferramentas de desenvolvimento e notebook. O lock boosting
acrescenta os dois modelos de comparação. Foram instaladas somente dependências Python
do projeto. Não foram instalados Colab CLI, GitHub CLI, WSL ou serviços pagos.

Use `scripts/diagnose.ps1` para repetir a coleta básica. Cada experimento também salva
versões, hardware observado e parâmetros de CPU/GPU. RAM livre e duração precisam ser
medidas novamente antes de um treino grande; não existe estimativa de treino oficial
enquanto os CSVs correspondentes não estiverem disponíveis.
