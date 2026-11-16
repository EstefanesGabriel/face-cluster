import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_PROJETO = Path(__file__).resolve().parent
load_dotenv(PASTA_PROJETO / ".env")


def _resolver_caminho(variavel: str, padrao: str) -> Path:
    valor = os.getenv(variavel, padrao).strip()
    caminho = Path(valor).expanduser()
    if not caminho.is_absolute():
        caminho = PASTA_PROJETO / caminho
    return caminho.resolve()


def _ler_inteiro_nao_negativo(variavel: str, padrao: int = 0) -> int:
    valor = os.getenv(variavel, str(padrao)).strip()
    if not valor:
        return padrao
    numero = int(valor)
    if numero < 0:
        raise ValueError(f"A variável {variavel} não pode ser negativa: {numero}")
    return numero


def _ler_float_intervalo(variavel: str, padrao: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    valor = os.getenv(variavel, str(padrao)).strip()
    if not valor:
        return padrao
    numero = float(valor)
    if not (minimo <= numero <= maximo):
        raise ValueError(f"A variável {variavel} deve estar entre {minimo} e {maximo}: {numero}")
    return numero


PASTA_IMAGENS = _resolver_caminho("INPUT_DIR", "./images")
PASTA_MODELOS = PASTA_PROJETO / "models"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

ARQUIVO_FACES = PASTA_RESULTADOS / "faces.json"
ARQUIVO_ROSTOS_ALINHADOS = PASTA_RESULTADOS / "rostos_alinhados.npy"
ARQUIVO_EMBEDDINGS = PASTA_RESULTADOS / "embeddings.npy"
ARQUIVO_CLUSTERS = PASTA_RESULTADOS / "clusters.json"
ARQUIVO_HTML_RESULTADO = PASTA_RESULTADOS / "resultado.html"

ETAPA_EXECUCAO = os.getenv("STEP", "todas").strip().lower()
LIMITE_IMAGENS = _ler_inteiro_nao_negativo("MAX_IMAGES", 0)
LIMIAR_SIMILARIDADE = _ler_float_intervalo("SIMILARITY_THRESHOLD", 0.52)

EXTENSOES_SUPORTADAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
