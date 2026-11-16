# Face Cluster

Pipeline modular em 4 etapas para detecção e alinhamento facial (SCRFD 2.5G ONNX), extração de embeddings (SFace ONNX), clusterização (HDBSCAN) e visualização em HTML sem duplicar imagens no disco.

## 1. Execução Rápida

Com Python instalado, execute no PowerShell ou Prompt de Comando:

```powershell
.\run.ps1
```
*(Ou dê dois cliques / execute `run.bat`)*

---

## 2. Configurando as Etapas e Limite de Imagens (`.env`)

Edite o arquivo `.env` na raiz do projeto para escolher a pasta de entrada, qual etapa deseja rodar (`STEP`) e o limite opcional de imagens (`MAX_IMAGES`):

```ini
# Pasta contendo as imagens de entrada (dataset)
INPUT_DIR=C:\Dev\celeba\input-clustering

# Etapa a ser executada:
# 1 ou detectar    = Etapa 1: Detecta e alinha as faces (salva resultados/faces.json e resultados/rostos_alinhados.npy)
# 2 ou embedding   = Etapa 2: Gera os vetores descritores 128D (salva resultados/embeddings.npy)
# 3 ou clusterizar = Etapa 3: Agrupa os embeddings (salva os caminhos em resultados/clusters.json)
# 4 ou resultado   = Etapa 4: Gera e abre resultados/resultado.html no navegador
# todas            = Executa as 4 etapas em sequência
STEP=todas

# Limite máximo de imagens a ler na Etapa 1 (0 = sem limite, processa todas da pasta)
MAX_IMAGES=0
```

Você também pode passar a etapa diretamente via linha de comando:

```powershell
.\.venv\Scripts\python.exe main.py detectar
.\.venv\Scripts\python.exe main.py embedding
.\.venv\Scripts\python.exe main.py clusterizar
.\.venv\Scripts\python.exe main.py resultado
.\.venv\Scripts\python.exe main.py todas
```

---

## 3. Arquivos Leves Gerados em `resultados/`

Todos os artefatos são salvos na pasta `resultados/` (ignorada pelo Git), sem copiar as fotos originais:

- **`resultados/faces.json`**: metadados das imagens com exatamente 1 face válida detectada.
- **`resultados/rostos_alinhados.npy`**: recortes `112x112x3` (`uint8`) alinhados pelos 5 pontos faciais, salvos sem perdas (*lossless*) e lidos via *memory-mapping* na Etapa 2.
- **`resultados/embeddings.npy`**: matriz `(N, 128)` (`float32`) de vetores descritores faciais.
- **`resultados/clusters.json`**: grupos gerados pelo HDBSCAN contendo apenas os caminhos para as imagens originais.
- **`resultados/resultado.html`**: página leve com os clusters bem separados exibindo as fotos diretamente de seus caminhos originais (`file:///`).
