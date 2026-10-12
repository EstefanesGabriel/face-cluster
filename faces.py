from pathlib import Path

import cv2
import numpy as np


PASTA_MODELOS = Path(__file__).resolve().parent / "models"

def carregar_modelos():
    detector = cv2.FaceDetectorYN.create(
        model=str(PASTA_MODELOS / "face_detection_yunet_2026may.onnx"),
        config="",
        input_size=(320, 320),
        score_threshold=0.7,
        nms_threshold=0.3,
    )

    extrator = cv2.FaceRecognizerSF.create(
        model=str(PASTA_MODELOS / "face_recognition_sface_2021dec.onnx"),
        config="",
    )

    return detector, extrator

def processar_imagem(caminho, detector, extrator):
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho}")

    altura, largura = imagem.shape[:2]
    detector.setInputSize((largura,altura))

    _, faces = detector.detect(imagem)

    if faces is None:
        return []

    resultados = []

    for face in faces:
        rosto_alinhado = extrator.alignCrop(imagem, face)
        embedding = extrator.feature(rosto_alinhado).flatten()

        resultados.append({
            "coordenadas": face[:4].tolist(),
            "confianca": float(face[-1]),
            "embedding": np.array(embedding, copy=True),
            "rosto_alinhado": rosto_alinhado.copy(),
        })

    return resultados