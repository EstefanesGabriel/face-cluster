import gc
import sys

from clustering import executar_clusterizacao
from config import ETAPA_EXECUCAO
from export import executar_etapa_resultado
from faces import executar_etapa_deteccao, executar_etapa_embeddings

MAPA_ETAPAS = {
    "1": "detectar",
    "detectar": "detectar",
    "2": "embedding",
    "embedding": "embedding",
    "embeddings": "embedding",
    "3": "clusterizar",
    "clusterizar": "clusterizar",
    "4": "resultado",
    "resultado": "resultado",
    "todas": "todas",
    "all": "todas",
}


def resolver_etapa() -> str:
    bruta = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ETAPA_EXECUCAO
    if bruta not in MAPA_ETAPAS:
        opcoes = "1/detectar, 2/embedding, 3/clusterizar, 4/resultado ou todas"
        raise ValueError(f"Etapa inválida '{bruta}'. Escolha entre: {opcoes}.")
    return MAPA_ETAPAS[bruta]


def executar_pipeline(etapa: str) -> None:
    if etapa == "detectar":
        executar_etapa_deteccao()
        return

    if etapa == "embedding":
        executar_etapa_embeddings()
        return

    if etapa == "clusterizar":
        executar_clusterizacao()
        return

    if etapa == "resultado":
        executar_etapa_resultado()
        return

    if not executar_etapa_deteccao():
        return
    gc.collect()

    if not executar_etapa_embeddings():
        return
    gc.collect()

    if not executar_clusterizacao():
        return
    gc.collect()

    executar_etapa_resultado()


def main() -> None:
    etapa = resolver_etapa()
    executar_pipeline(etapa)


if __name__ == "__main__":
    main()
