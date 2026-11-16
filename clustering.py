import json

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from config import (
    ARQUIVO_CLUSTERS,
    ARQUIVO_EMBEDDINGS,
    ARQUIVO_FACES,
    LIMIAR_SIMILARIDADE,
    PASTA_RESULTADOS,
)


def validar_embeddings(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Os embeddings devem formar uma matriz.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Existem embeddings com valores inválidos.")
    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Existem embeddings com todos os valores iguais a zero.")


def refinar_clusters_hdbscan(
    embeddings_normalizados: np.ndarray,
    rotulos: np.ndarray,
    limiar_similaridade: float,
) -> np.ndarray:
    """Remove membros de baixa coesão e dissolve agrupamentos mal definidos do HDBSCAN."""
    rotulos_refinados = rotulos.copy()

    for grupo in np.unique(rotulos_refinados[rotulos_refinados >= 0]):
        indices = np.flatnonzero(rotulos_refinados == grupo)

        while len(indices) >= 2:
            sub = embeddings_normalizados[indices]
            matriz_sim = sub @ sub.T
            np.fill_diagonal(matriz_sim, 1.0)
            medias = (matriz_sim.sum(axis=1) - 1.0) / (len(indices) - 1)
            pior = int(np.argmin(medias))

            if medias[pior] < limiar_similaridade:
                rotulos_refinados[indices[pior]] = -1
                indices = np.delete(indices, pior)
            else:
                break

        if len(indices) < 2:
            rotulos_refinados[indices] = -1

    return rotulos_refinados


def organizar_e_numerar_grupos(rotulos: np.ndarray) -> tuple[np.ndarray, int]:
    """Ordena clusters válidos (do maior para o menor) e converte ruídos (-1) em grupos individuais."""
    rotulos_finais = np.full_like(rotulos, -1)
    grupos_validos = [
        np.flatnonzero(rotulos == g)
        for g in np.unique(rotulos[rotulos >= 0])
    ]
    grupos_validos.sort(key=lambda idxs: (-len(idxs), int(idxs[0])))

    proximo_id = 0
    for membros in grupos_validos:
        rotulos_finais[membros] = proximo_id
        proximo_id += 1

    indices_individuais = np.flatnonzero(rotulos == -1)
    for indice in indices_individuais:
        rotulos_finais[indice] = proximo_id
        proximo_id += 1

    return rotulos_finais, len(indices_individuais)


def agrupar_faces(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    validar_embeddings(embeddings)

    if len(embeddings) < 2:
        return np.arange(len(embeddings)), len(embeddings)

    embeddings_normalizados = normalize(embeddings, norm="l2")

    modelo = HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="leaf",
        copy=True,
    )

    rotulos_brutos = modelo.fit_predict(embeddings_normalizados)
    rotulos_refinados = refinar_clusters_hdbscan(
        embeddings_normalizados,
        rotulos_brutos,
        LIMIAR_SIMILARIDADE,
    )
    return organizar_e_numerar_grupos(rotulos_refinados)


def exibir_resumo_grupos(
    total_faces: int,
    rotulos: np.ndarray,
    quantidade_individuais: int,
) -> int:
    grupos, quantidades = np.unique(rotulos, return_counts=True)
    grupos_multiplos = len(grupos) - quantidade_individuais

    print(f"Faces processadas: {total_faces}")
    print(f"Limiar de coesão (SIMILARITY_THRESHOLD): {LIMIAR_SIMILARIDADE:.2f}")
    print(f"Clusters validados pelo HDBSCAN (2+ faces): {grupos_multiplos}")
    print(f"Faces individuais (1 face): {quantidade_individuais}")
    print(f"Total de grupos na saída: {len(grupos)}")

    for grupo, quantidade in zip(grupos, quantidades):
        if quantidade > 1:
            print(f"Cluster {grupo:03d}: {quantidade} face(s)")

    return len(grupos)


def salvar_clusters(
    registros: list[dict],
    rotulos: np.ndarray,
    quantidade_individuais: int,
) -> dict:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    registros_por_indice = {int(r["indice_embedding"]): r for r in registros}

    if sorted(registros_por_indice.keys()) != list(range(len(rotulos))):
        raise ValueError("Os registros de faces.json não correspondem aos rótulos de embeddings.")

    grupos_saida = []
    for grupo in np.unique(rotulos):
        membros = [int(i) for i in np.flatnonzero(rotulos == grupo)]
        imagens = [registros_por_indice[i]["imagem"] for i in membros]
        grupos_saida.append({
            "grupo": int(grupo),
            "quantidade": len(imagens),
            "indices": membros,
            "imagens": imagens,
        })

    dados_clusters = {
        "total_faces": len(registros),
        "total_grupos": len(grupos_saida),
        "grupos_hdbscan": len(grupos_saida) - quantidade_individuais,
        "grupos_individuais": quantidade_individuais,
        "grupos": grupos_saida,
    }

    with ARQUIVO_CLUSTERS.open("w", encoding="utf-8") as arquivo:
        json.dump(dados_clusters, arquivo, ensure_ascii=False, indent=2)

    return dados_clusters


def executar_clusterizacao() -> bool:
    """Etapa 3: Executa o agrupamento HDBSCAN com poda de coesão e salva em resultados/clusters.json."""
    print("\n=== Etapa 3: Clusterização (HDBSCAN) ===")
    if not ARQUIVO_EMBEDDINGS.is_file() or not ARQUIVO_FACES.is_file():
        print(f"Arquivos de entrada não encontrados em {PASTA_RESULTADOS}.")
        print("Execute primeiro as Etapas 1 e 2.")
        return False

    embeddings = np.load(ARQUIVO_EMBEDDINGS, allow_pickle=False)
    with ARQUIVO_FACES.open("r", encoding="utf-8") as arquivo:
        registros = json.load(arquivo)

    rotulos, quantidade_individuais = agrupar_faces(embeddings)
    exibir_resumo_grupos(len(embeddings), rotulos, quantidade_individuais)
    salvar_clusters(registros, rotulos, quantidade_individuais)

    print(f"\nClusters salvos (sem cópia de imagens) em: {ARQUIVO_CLUSTERS}")
    return True


def main() -> None:
    executar_clusterizacao()


if __name__ == "__main__":
    main()
