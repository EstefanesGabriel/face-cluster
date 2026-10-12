from pathlib import Path
import cv2
from faces import carregar_modelos, processar_imagem
import json
import numpy as np


PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_IMAGENS = PASTA_PROJETO / "images"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PASTA_DADOS = PASTA_PROJETO / "data"

PROCESSAR_TODAS = True

IMAGEM_ESPECIFICA = Path(
    r"C:\Gabriel\utfpr\face-cluster\images\Celebrity Faces Dataset\195_dc745cfd.jpg"
)


def main():
    if PROCESSAR_TODAS:
        caminhos = sorted(
            caminho
            for caminho in PASTA_IMAGENS.rglob("*")
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES
        )
    else:
        if not IMAGEM_ESPECIFICA.is_file():
            print(f"Imagem não encontrada: {IMAGEM_ESPECIFICA}")
            return

        caminhos = [IMAGEM_ESPECIFICA]

    if not caminhos:
        print("Nenhuma imagem encontrada.")
        return

    detector, extrator = carregar_modelos()

    registros = []
    embeddings = []

    for numero, caminho in enumerate(caminhos, start=1):
        # Usa caminho relativo quando a imagem está dentro do projeto.
        try:
            origem = caminho.relative_to(PASTA_PROJETO).as_posix()
        except ValueError:
            origem = caminho.as_posix()

        try:
            resultados = processar_imagem(caminho, detector, extrator)
        except (ValueError, cv2.error) as erro:
            print(f"[{numero}/{len(caminhos)}] Erro em {origem}: {erro}")
            continue

        for numero_face, resultado in enumerate(resultados, start=1):
            registros.append({
                "indice_embedding": len(embeddings),
                "imagem": origem,
                "coordenadas": resultado["coordenadas"],
                "confianca": resultado["confianca"],
            })

            embeddings.append(resultado["embedding"])

            if not PROCESSAR_TODAS:
                print(
                    f"Face {numero_face}: "
                    f"escore {resultado['confianca']:.3f}"
                )

                cv2.imshow(
                    f"Face {numero_face}",
                    resultado["rosto_alinhado"],
                )

        print(
            f"[{numero}/{len(caminhos)}] "
            f"{origem}: {len(resultados)} face(s)"
        )

    print(f"\nTotal de faces detectadas: {len(registros)}")
    print(f"Total de embeddings: {len(embeddings)}")

    if not PROCESSAR_TODAS and registros:
        print("\nPressione uma tecla na janela da imagem para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if PROCESSAR_TODAS:
        if not embeddings:
            print("\nNenhuma face encontrada. Os arquivos não foram atualizados.")
            return

        PASTA_DADOS.mkdir(parents=True, exist_ok=True)

        matriz_embeddings = np.stack(embeddings).astype(np.float32)

        np.save(
            PASTA_DADOS / "embeddings.npy",
            matriz_embeddings,
            allow_pickle=False,
        )

        with (PASTA_DADOS / "faces.json").open(
            "w",
            encoding="utf-8",
        ) as arquivo:
            json.dump(
                registros,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nDados salvos em: {PASTA_DADOS}")
        print(f"Formato dos embeddings: {matriz_embeddings.shape}")


if __name__ == "__main__":
    main()