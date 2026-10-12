from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
import json
from export import exportar_faces


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_EMBEDDINGS = PASTA_PROJETO / "data" / "embeddings.npy"


def agrupar_faces(embeddings):
    if embeddings.ndim != 2:
        raise ValueError("Os embeddings devem formar uma matriz.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Existem embeddings com valores inválidos.")

    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Existem embeddings com todos os valores iguais a zero.")

    # Sem duas faces, não há agrupamento a realizar.
    if len(embeddings) < 2:
        return np.arange(len(embeddings)), len(embeddings)

    # Coloca cada vetor na mesma escala, preservando sua direção.
    embeddings_normalizados = normalize(embeddings, norm="l2")

    modelo = HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    rotulos = modelo.fit_predict(embeddings_normalizados)

    # Cada face sem atribuição recebe um grupo exclusivo.
    indices_sem_grupo = np.flatnonzero(rotulos == -1)
    proximo_grupo = int(rotulos.max()) + 1

    for indice in indices_sem_grupo:
        rotulos[indice] = proximo_grupo
        proximo_grupo += 1

    return rotulos, len(indices_sem_grupo)


def main():
    if not ARQUIVO_EMBEDDINGS.is_file():
        print("Arquivo embeddings.npy não encontrado.")
        print("Execute primeiro o processamento de todas as imagens.")
        return

    embeddings = np.load(ARQUIVO_EMBEDDINGS, allow_pickle=False)

    arquivo_faces = PASTA_PROJETO / "data" / "faces.json"

    with arquivo_faces.open("r", encoding="utf-8") as arquivo:
        registros = json.load(arquivo)

    rotulos, quantidade_individuais = agrupar_faces(embeddings)

    grupos, quantidades = np.unique(rotulos, return_counts=True)

    print(f"Faces processadas: {len(embeddings)}")
    print(f"Grupos encontrados pelo HDBSCAN: {len(grupos) - quantidade_individuais}")
    print(f"Grupos individuais adicionados: {quantidade_individuais}")
    print(f"Total de grupos na saída: {len(grupos)}")

    for grupo, quantidade in zip(grupos, quantidades):
        print(f"Grupo {grupo:03d}: {quantidade} face(s)")

    pasta_saida = PASTA_PROJETO / "output"

    quantidade_salva = exportar_faces(
        registros,
        rotulos,
        PASTA_PROJETO,
        pasta_saida,
        embeddings,
    )

    print(f"\nRostos representativos salvos: {len(grupos)}")
    print(f"Fotos originais copiadas: {quantidade_salva}")
    print(f"Pasta de saída: {pasta_saida}")


if __name__ == "__main__":
    main()
