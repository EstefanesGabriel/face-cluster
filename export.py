import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def escolher_representante(indices, embeddings):
    """Escolhe o membro com maior similaridade cosseno média com o grupo."""
    vetores = np.asarray(embeddings[indices], dtype=np.float64)
    vetores = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
    # Equivale a comparar todos os pares, sem criar uma matriz N x N.
    similaridades = vetores @ vetores.sum(axis=0)
    return indices[int(np.argmax(similaridades))]


def salvar_rosto_representativo(registro, caminho, destino):
    # imdecode também aceita caminhos com acentos no Windows.
    imagem = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {caminho}")

    altura, largura = imagem.shape[:2]
    x, y, w, h = registro["coordenadas"]
    esquerda = max(0, math.floor(x))
    topo = max(0, math.floor(y))
    direita = min(largura, math.ceil(x + w))
    base = min(altura, math.ceil(y + h))
    if direita <= esquerda or base <= topo:
        raise ValueError(f"Recorte inválido para a face {registro['indice_embedding']}.")

    sucesso, dados = cv2.imencode(".jpg", imagem[topo:base, esquerda:direita])
    if not sucesso:
        raise OSError(f"Não foi possível gerar: {destino}")
    dados.tofile(destino)


def exportar_faces(registros, rotulos, pasta_projeto, pasta_saida, embeddings):
    pasta_projeto = Path(pasta_projeto).resolve()
    pasta_saida = Path(pasta_saida).resolve()
    # Apenas esta pasta gerada pode ser substituída; nunca images ou o projeto.
    if pasta_saida != pasta_projeto / "output":
        raise ValueError("A exportação deve usar a pasta output na raiz do projeto.")

    rotulos = np.asarray(rotulos)
    embeddings = np.asarray(embeddings)
    indices = [registro["indice_embedding"] for registro in registros]
    if rotulos.ndim != 1 or sorted(indices) != list(range(len(rotulos))):
        raise ValueError("Os registros de faces.json não correspondem aos rótulos.")
    if len(rotulos) == 0:
        raise ValueError("Nenhuma face para exportar. A saída anterior foi preservada.")
    if not np.issubdtype(rotulos.dtype, np.integer) or np.any(rotulos < 0):
        raise ValueError("Cada face deve ter um identificador de grupo não negativo.")
    if (embeddings.ndim != 2 or len(embeddings) != len(rotulos)
            or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()
            or np.any(np.linalg.norm(embeddings, axis=1) == 0)):
        raise ValueError("Embeddings inválidos ou incompatíveis com os registros.")

    registros_por_indice = {r["indice_embedding"]: r for r in registros}
    caminhos = {}
    for indice, registro in registros_por_indice.items():
        caminho = Path(registro["imagem"])
        if not caminho.is_absolute():
            caminho = pasta_projeto / caminho
        caminho = caminho.resolve()
        if caminho.is_relative_to(pasta_saida):
            raise ValueError("As imagens de origem não podem estar dentro de output.")
        if not caminho.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
        caixa = np.asarray(registro["coordenadas"], dtype=float)
        if caixa.shape != (4,) or not np.isfinite(caixa).all() or np.any(caixa[2:] <= 0):
            raise ValueError(f"Coordenadas inválidas para a face {indice}.")
        caminhos[indice] = caminho

    # Prepara tudo antes de substituir output. Falhas no preparo preservam a saída.
    pasta_temporaria = Path(tempfile.mkdtemp(prefix=".exportacao-", dir=pasta_projeto)).resolve()
    if pasta_temporaria.parent != pasta_projeto:
        raise ValueError("Pasta temporária fora do projeto.")
    anterior = pasta_temporaria / "anterior"
    publicada = False
    try:
        nova_saida = pasta_temporaria / "nova"
        nova_saida.mkdir()
        quantidade_fotos = 0

        for grupo in np.unique(rotulos):
            membros = np.flatnonzero(rotulos == grupo)
            representante = escolher_representante(membros, embeddings)
            pasta_grupo = nova_saida / f"pessoa_{grupo:03d}"
            pasta_grupo.mkdir()
            salvar_rosto_representativo(
                registros_por_indice[representante], caminhos[representante],
                pasta_grupo / "rosto_representativo.jpg",
            )

            # Duas faces do mesmo grupo na mesma foto não duplicam o arquivo.
            originais = sorted({caminhos[int(indice)] for indice in membros})
            for numero, origem in enumerate(originais, start=1):
                destino = pasta_grupo / f"foto_{numero:04d}_{origem.name}"
                shutil.copy2(origem, destino)
                quantidade_fotos += 1

        # Mantém output no lugar: o Explorer pode manter a pasta aberta no Windows.
        pasta_saida.mkdir(exist_ok=True)
        anterior.mkdir()
        movidos_antigos = []
        movidos_novos = []
        try:
            for item in list(pasta_saida.iterdir()):
                item.rename(anterior / item.name)
                movidos_antigos.append(item.name)
            for item in list(nova_saida.iterdir()):
                item.rename(pasta_saida / item.name)
                movidos_novos.append(item.name)
        except OSError:
            for nome in reversed(movidos_novos):
                (pasta_saida / nome).rename(nova_saida / nome)
            for nome in reversed(movidos_antigos):
                (anterior / nome).rename(pasta_saida / nome)
            raise
        publicada = True
    finally:
        # Se a restauração também falhar, mantém o backup para recuperação.
        if publicada or not anterior.exists() or not any(anterior.iterdir()):
            shutil.rmtree(pasta_temporaria)

    return quantidade_fotos
