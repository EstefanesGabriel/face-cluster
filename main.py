from pathlib import Path

from faces import processar_imagem


PASTA_PROJETO = Path(__file__).resolve().parent


def main():
    caminho = (
        PASTA_PROJETO
        / "images"
        / "Celebrity Faces Dataset"
        / "001_08194468.jpg"
    )

    resultados = processar_imagem(caminho)

    print(f"Imagem: {caminho.name}")
    print(f"Faces detectadas: {len(resultados)}")

    for numero, resultado in enumerate(resultados, start=1):
        print(f"\nFace {numero}")
        print(f"Coordenadas: {resultado['coordenadas']}")
        print(f"Escore de detecção: {resultado['confianca']:.3f}")
        print(f"Tamanho do embedding: {resultado['embedding'].shape}")


if __name__ == "__main__":
    main()