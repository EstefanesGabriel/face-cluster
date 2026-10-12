# Face Cluster

Agrupamento de rostos em fotos usando OpenCV (YuNet e SFace) e HDBSCAN.

## 1. Preparar o ambiente

Com Python instalado, abra o PowerShell na pasta do projeto. O ambiente usado no desenvolvimento é Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python==5.0.0.93 numpy==2.5.4 scikit-learn==1.9.1
```

Os comandos abaixo usam diretamente o Python do ambiente virtual, sem precisar ativá-lo.

## 2. Adicionar os modelos e as imagens

Baixe os modelos no OpenCV Zoo e coloque os arquivos `.onnx` na pasta `models/`, mantendo estes nomes:

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): `face_detection_yunet_2026may.onnx`.
- [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface): `face_recognition_sface_2021dec.onnx`.

No GitHub, abra o arquivo e use **Download raw file** para baixar o modelo.

Coloque suas fotos em `images/`. Subpastas também são processadas. Os formatos aceitos são JPG, JPEG, PNG, BMP e WEBP.

## 3. Processar os rostos

Mantenha `PROCESSAR_TODAS = True` em `main.py` e execute:

```powershell
.\.venv\Scripts\python.exe main.py
```

Esse passo detecta os rostos e gera `data/faces.json` e `data/embeddings.npy`. Quando há faces detectadas, os arquivos anteriores são sobrescritos.

## 4. Agrupar e exportar

```powershell
.\.venv\Scripts\python.exe clustering.py
```

Os resultados ficam em `output/pessoa_000/`, `output/pessoa_001/` e assim por diante. Cada pasta contém um rosto representativo e cópias das fotos originais completas. Faces sem grupo atribuído pelo HDBSCAN ficam em pastas individuais.

A cada exportação concluída, o conteúdo anterior de `output/` é substituído. Para testar outros parâmetros de clusterização, basta executar `clustering.py` novamente. Se mudar as imagens ou os modelos, execute primeiro `main.py`.

## Testar uma imagem

Em `main.py`, defina `PROCESSAR_TODAS = False` e ajuste `IMAGEM_ESPECIFICA` para o caminho da foto. Execute `main.py` para visualizar os rostos alinhados. Nesse modo, os arquivos de `data/` não são atualizados.
