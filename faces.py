import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from config import (
    ARQUIVO_EMBEDDINGS,
    ARQUIVO_FACES,
    ARQUIVO_ROSTOS_ALINHADOS,
    EXTENSOES_SUPORTADAS,
    LIMITE_IMAGENS,
    PASTA_IMAGENS,
    PASTA_MODELOS,
    PASTA_PROJETO,
    PASTA_RESULTADOS,
)

MODELO_DETECTOR = "det_2.5g.onnx"
MODELO_EXTRATOR = "face_recognition_sface_2021dec.onnx"

URLS_MODELOS = {
    MODELO_DETECTOR: [
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
        "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_2.5g.onnx",
        "https://huggingface.co/ArjunSharmaSSS/vision-detection-models/resolve/main/det_2.5g.onnx",
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx",
    ],
    MODELO_EXTRATOR: [
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}

PONTOS_REFERENCIA_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

TAMANHO_MINIMO_FACE = 30


def garantir_modelo(nome_arquivo: str) -> Path:
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    caminho_modelo = PASTA_MODELOS / nome_arquivo
    if caminho_modelo.is_file():
        return caminho_modelo

    urls = URLS_MODELOS[nome_arquivo]
    caminho_temp = caminho_modelo.with_suffix(".onnx.part")
    ultimo_erro = None

    print(f"Baixando modelo {nome_arquivo}...")
    for url in urls:
        try:
            requisicao = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(requisicao, timeout=60) as resposta, caminho_temp.open("wb") as destino:
                while bloco := resposta.read(65536):
                    destino.write(bloco)
            caminho_temp.replace(caminho_modelo)
            return caminho_modelo
        except (URLError, OSError) as erro:
            ultimo_erro = erro
            if caminho_temp.exists():
                caminho_temp.unlink(missing_ok=True)

    raise RuntimeError(f"Não foi possível baixar o modelo {nome_arquivo}: {ultimo_erro}")


class DetectorSCRFD:
    """Detector facial SCRFD 2.5G com 5 pontos faciais via OpenCV DNN."""

    def __init__(
        self,
        caminho_modelo: Path,
        tamanho_entrada: tuple[int, int] = (640, 640),
        limiar_confianca: float = 0.5,
        limiar_nms: float = 0.4,
    ) -> None:
        self.rede = cv2.dnn.readNetFromONNX(str(caminho_modelo))
        self.nomes_saida = self.rede.getUnconnectedOutLayersNames()
        self.tamanho_entrada = tamanho_entrada
        self.limiar_confianca = limiar_confianca
        self.limiar_nms = limiar_nms
        self.strides = (8, 16, 32)
        self.num_ancoras = 2
        self._centros_ancoras = self._gerar_centros_ancoras()

    def _gerar_centros_ancoras(self) -> dict[int, np.ndarray]:
        largura_rede, altura_rede = self.tamanho_entrada
        centros = {}
        for stride in self.strides:
            altura_mapa = altura_rede // stride
            largura_mapa = largura_rede // stride
            grade = np.stack(np.mgrid[:altura_mapa, :largura_mapa][::-1], axis=-1).astype(np.float32)
            grade = (grade * stride).reshape((-1, 2))
            grade = np.repeat(grade, self.num_ancoras, axis=0)
            centros[stride] = grade
        return centros

    def _preparar_letterbox(self, imagem: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        altura_img, largura_img = imagem.shape[:2]
        largura_rede, altura_rede = self.tamanho_entrada

        escala = min(largura_rede / largura_img, altura_rede / altura_img)
        nova_largura = max(1, int(round(largura_img * escala)))
        nova_altura = max(1, int(round(altura_img * escala)))

        interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
        redimensionada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=interp)

        offset_x = (largura_rede - nova_largura) // 2
        offset_y = (altura_rede - nova_altura) // 2

        canvas = np.zeros((altura_rede, largura_rede, 3), dtype=np.uint8)
        canvas[offset_y : offset_y + nova_altura, offset_x : offset_x + nova_largura] = redimensionada
        return canvas, escala, offset_x, offset_y

    def detectar(self, imagem: np.ndarray) -> np.ndarray:
        altura_img, largura_img = imagem.shape[:2]
        canvas, escala, offset_x, offset_y = self._preparar_letterbox(imagem)

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=self.tamanho_entrada,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        self.rede.setInput(blob)
        saidas = self.rede.forward(self.nomes_saida)

        tensores_por_formato = {}
        for saida in saidas:
            matriz = np.squeeze(saida)
            if matriz.ndim == 1:
                matriz = matriz[:, np.newaxis]
            tensores_por_formato[matriz.shape] = matriz

        lista_caixas = []
        lista_pontos = []
        lista_escores = []

        largura_rede, altura_rede = self.tamanho_entrada
        for stride in self.strides:
            total_ancoras = (altura_rede // stride) * (largura_rede // stride) * self.num_ancoras
            escores = tensores_por_formato[(total_ancoras, 1)].ravel()
            caixas_dist = tensores_por_formato[(total_ancoras, 4)] * stride
            pontos_dist = tensores_por_formato[(total_ancoras, 10)] * stride

            mascara = escores >= self.limiar_confianca
            if not np.any(mascara):
                continue

            centros = self._centros_ancoras[stride][mascara]
            escores_filtrados = escores[mascara]
            caixas_filtradas = caixas_dist[mascara]
            pontos_filtrados = pontos_dist[mascara]

            x1 = (centros[:, 0] - caixas_filtradas[:, 0] - offset_x) / escala
            y1 = (centros[:, 1] - caixas_filtradas[:, 1] - offset_y) / escala
            x2 = (centros[:, 0] + caixas_filtradas[:, 2] - offset_x) / escala
            y2 = (centros[:, 1] + caixas_filtradas[:, 3] - offset_y) / escala

            x1 = np.clip(x1, 0.0, float(largura_img))
            y1 = np.clip(y1, 0.0, float(altura_img))
            x2 = np.clip(x2, 0.0, float(largura_img))
            y2 = np.clip(y2, 0.0, float(altura_img))

            caixas_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)

            pontos = np.empty_like(pontos_filtrados)
            pontos[:, 0::2] = (centros[:, 0:1] + pontos_filtrados[:, 0::2] - offset_x) / escala
            pontos[:, 1::2] = (centros[:, 1:2] + pontos_filtrados[:, 1::2] - offset_y) / escala

            lista_caixas.append(caixas_xywh)
            lista_pontos.append(pontos)
            lista_escores.append(escores_filtrados)

        if not lista_escores:
            return np.empty((0, 15), dtype=np.float32)

        caixas = np.vstack(lista_caixas).astype(np.float32)
        pontos = np.vstack(lista_pontos).astype(np.float32)
        escores = np.concatenate(lista_escores).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=caixas.tolist(),
            scores=escores.tolist(),
            score_threshold=self.limiar_confianca,
            nms_threshold=self.limiar_nms,
        )
        if len(indices) == 0:
            return np.empty((0, 15), dtype=np.float32)

        indices = np.asarray(indices).ravel()
        faces = np.hstack([caixas[indices], pontos[indices], escores[indices, np.newaxis]])

        mascara_tamanho = (faces[:, 2] >= TAMANHO_MINIMO_FACE) & (faces[:, 3] >= TAMANHO_MINIMO_FACE)
        return faces[mascara_tamanho].astype(np.float32)


def carregar_detector() -> DetectorSCRFD:
    caminho = garantir_modelo(MODELO_DETECTOR)
    return DetectorSCRFD(caminho)


def carregar_extrator():
    caminho = garantir_modelo(MODELO_EXTRATOR)
    return cv2.FaceRecognizerSF.create(model=str(caminho), config="")


def ler_imagem(caminho: Path) -> np.ndarray:
    dados = np.fromfile(caminho, dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho}")
    return imagem


def alinhar_rosto(imagem: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Alinha o rosto em 112x112 diretamente a partir da imagem original em resolução máxima."""
    pontos_origem = face[4:14].reshape(5, 2).astype(np.float32)
    matriz, _ = cv2.estimateAffinePartial2D(
        pontos_origem,
        PONTOS_REFERENCIA_112,
        method=cv2.LMEDS,
    )
    if matriz is None:
        x, y, w, h = [max(0, int(round(v))) for v in face[:4]]
        recorte = imagem[y : y + max(1, h), x : x + max(1, w)]
        return cv2.resize(recorte, (112, 112))
    return cv2.warpAffine(imagem, matriz, (112, 112))


def listar_imagens(pasta_imagens: Path, limite: int = 0) -> list[Path]:
    if not pasta_imagens.is_dir():
        return []

    caminhos: list[Path] = []
    pilha = [pasta_imagens]

    while pilha:
        atual = pilha.pop()
        subpastas = []
        arquivos = []

        with os.scandir(atual) as entradas:
            for entrada in entradas:
                if entrada.is_dir(follow_symlinks=False):
                    subpastas.append(Path(entrada.path))
                elif entrada.is_file(follow_symlinks=False):
                    sufixo = os.path.splitext(entrada.name)[1].lower()
                    if sufixo in EXTENSOES_SUPORTADAS:
                        arquivos.append(Path(entrada.path))
                        if limite > 0 and not subpastas and len(caminhos) + len(arquivos) >= limite:
                            break

        arquivos.sort()
        caminhos.extend(arquivos)
        if limite > 0 and len(caminhos) >= limite:
            return caminhos[:limite]

        for subpasta in sorted(subpastas, reverse=True):
            pilha.append(subpasta)

    return caminhos


def formatar_caminho_origem(caminho: Path) -> str:
    try:
        return caminho.relative_to(PASTA_PROJETO).as_posix()
    except ValueError:
        return caminho.resolve().as_posix()


def executar_etapa_deteccao() -> bool:
    """Etapa 1: Detecta e alinha faces (descartando imagens com != 1 face), salvando sem perdas."""
    print("\n=== Etapa 1: Detecção e Alinhamento de Faces ===")
    caminhos = listar_imagens(PASTA_IMAGENS, LIMITE_IMAGENS)
    if not caminhos:
        print(f"Nenhuma imagem encontrada em: {PASTA_IMAGENS}")
        return False

    if LIMITE_IMAGENS > 0:
        print(f"Limite configurado (MAX_IMAGES={LIMITE_IMAGENS}): processando {len(caminhos)} imagem(ns).")
    else:
        print(f"Processando {len(caminhos)} imagem(ns) de {PASTA_IMAGENS}...")

    detector = carregar_detector()
    registros = []
    rostos_alinhados = []
    descartadas_sem_face = 0
    descartadas_multiplas = 0

    for numero, caminho in enumerate(caminhos, start=1):
        origem = formatar_caminho_origem(caminho)

        try:
            imagem = ler_imagem(caminho)
            faces = detector.detectar(imagem)
        except (ValueError, cv2.error) as erro:
            print(f"[{numero}/{len(caminhos)}] Erro em {origem}: {erro}")
            continue

        if len(faces) == 0:
            descartadas_sem_face += 1
            print(f"[{numero}/{len(caminhos)}] {origem}: descartada (0 faces)")
            continue

        if len(faces) > 1:
            descartadas_multiplas += 1
            print(f"[{numero}/{len(caminhos)}] {origem}: descartada ({len(faces)} faces detectadas)")
            continue

        face = faces[0]
        rosto_112 = alinhar_rosto(imagem, face)

        registros.append({
            "indice_embedding": len(rostos_alinhados),
            "imagem": origem,
            "coordenadas": [round(float(v), 2) for v in face[:4]],
            "confianca": round(float(face[14]), 4),
        })
        rostos_alinhados.append(rosto_112)
        print(f"[{numero}/{len(caminhos)}] {origem}: 1 face alinhada (confiança {face[14]:.3f})")

    print("\n--- Resumo da Etapa 1 (Detecção e Alinhamento) ---")
    print(f"Imagens analisadas: {len(caminhos)}")
    print(f"Faces válidas alinhadas (1 por imagem): {len(registros)}")
    print(f"Imagens descartadas (0 faces): {descartadas_sem_face}")
    print(f"Imagens descartadas (>1 face): {descartadas_multiplas}")

    if not rostos_alinhados:
        print("Nenhuma face válida encontrada. Os arquivos em resultados/ não foram atualizados.")
        return False

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    matriz_rostos = np.stack(rostos_alinhados).astype(np.uint8)
    np.save(ARQUIVO_ROSTOS_ALINHADOS, matriz_rostos, allow_pickle=False)

    with ARQUIVO_FACES.open("w", encoding="utf-8") as arquivo:
        json.dump(registros, arquivo, ensure_ascii=False, indent=2)

    print(f"Metadados salvos em: {ARQUIVO_FACES}")
    print(f"Rostos alinhados (sem perdas) salvos em: {ARQUIVO_ROSTOS_ALINHADOS} {matriz_rostos.shape}")
    return True


def executar_etapa_embeddings() -> bool:
    """Etapa 2: Lê os rostos alinhados via memory-mapping e gera os vetores descritores (128D)."""
    print("\n=== Etapa 2: Geração de Embeddings (SFace) ===")
    if not ARQUIVO_ROSTOS_ALINHADOS.is_file() or not ARQUIVO_FACES.is_file():
        print(f"Arquivos da Etapa 1 não encontrados em {PASTA_RESULTADOS}.")
        print("Execute primeiro a Etapa 1 (STEP=detectar).")
        return False

    with ARQUIVO_FACES.open("r", encoding="utf-8") as arquivo:
        registros = json.load(arquivo)

    rostos_mmap = np.load(ARQUIVO_ROSTOS_ALINHADOS, mmap_mode="r", allow_pickle=False)
    if rostos_mmap.ndim != 4 or rostos_mmap.shape[1:] != (112, 112, 3) or len(rostos_mmap) != len(registros):
        raise ValueError("O arquivo rostos_alinhados.npy está incompatível com faces.json.")

    extrator = carregar_extrator()
    total = len(rostos_mmap)
    matriz_embeddings = np.empty((total, 128), dtype=np.float32)

    for indice in range(total):
        rosto = np.asarray(rostos_mmap[indice], dtype=np.uint8)
        matriz_embeddings[indice] = extrator.feature(rosto).flatten()

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    np.save(ARQUIVO_EMBEDDINGS, matriz_embeddings, allow_pickle=False)

    print(f"Embeddings extraídos com sucesso: {matriz_embeddings.shape}")
    print(f"Arquivo salvo em: {ARQUIVO_EMBEDDINGS}")
    return True
