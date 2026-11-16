import base64
import html
import json
import webbrowser
from pathlib import Path

import cv2
import numpy as np

from config import (
    ARQUIVO_CLUSTERS,
    ARQUIVO_HTML_RESULTADO,
    ARQUIVO_ROSTOS_ALINHADOS,
    PASTA_PROJETO,
    PASTA_RESULTADOS,
)


def _resolver_caminho_imagem(caminho_str: str) -> Path:
    caminho = Path(caminho_str)
    if not caminho.is_absolute():
        caminho = PASTA_PROJETO / caminho
    return caminho.resolve()


def _codificar_rosto_png_base64(rosto_bgr: np.ndarray) -> str:
    sucesso, buffer = cv2.imencode(".png", rosto_bgr)
    if not sucesso:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def gerar_html_clusters(dados_clusters: dict, rostos_alinhados: np.ndarray | None) -> str:
    total_faces = int(dados_clusters.get("total_faces", 0))
    total_grupos = int(dados_clusters.get("total_grupos", 0))
    grupos_hdbscan = int(dados_clusters.get("grupos_hdbscan", 0))
    grupos_individuais = int(dados_clusters.get("grupos_individuais", 0))
    grupos = dados_clusters.get("grupos", [])

    blocos_clusters = []
    for item in grupos:
        grupo_id = int(item["grupo"])
        quantidade = int(item["quantidade"])
        imagens = item.get("imagens", [])
        indices = item.get("indices", [])
        classe_individual = " cluster-individual" if quantidade == 1 else ""

        cartoes_imagens = []
        for pos, caminho_str in enumerate(imagens):
            caminho_absoluto = _resolver_caminho_imagem(caminho_str)
            uri_segura = html.escape(caminho_absoluto.as_uri(), quote=True)
            nome_seguro = html.escape(caminho_absoluto.name)

            bloco_alinhado = ""
            if rostos_alinhados is not None and pos < len(indices):
                indice_emb = int(indices[pos])
                if 0 <= indice_emb < len(rostos_alinhados):
                    data_uri = _codificar_rosto_png_base64(np.asarray(rostos_alinhados[indice_emb], dtype=np.uint8))
                    if data_uri:
                        bloco_alinhado = (
                            f'<div class="col-img img-alinhada">'
                            f'<img src="{data_uri}" alt="Alinhada {nome_seguro}" title="Rosto alinhado 112x112 (entrada do embedding)">'
                            f"<small>Alinhada 112x112</small>"
                            f"</div>"
                        )

            cartoes_imagens.append(
                f'<div class="foto">'
                f'<div class="par-imagens">'
                f'<div class="col-img img-original">'
                f'<a href="{uri_segura}" target="_blank">'
                f'<img src="{uri_segura}" alt="{nome_seguro}" loading="lazy" title="Imagem original">'
                f"</a>"
                f"<small>Original</small>"
                f"</div>"
                f"{bloco_alinhado}"
                f"</div>"
                f'<span class="nome-arquivo">{nome_seguro}</span>'
                f"</div>"
            )

        grade_html = "\n".join(cartoes_imagens)
        blocos_clusters.append(
            f'<section class="cluster{classe_individual}">'
            f"<h2>Cluster {grupo_id:03d} <small>({quantidade} imagem(ns))</small></h2>"
            f'<div class="grade">\n{grade_html}\n</div>'
            f"</section>"
        )

    conteudo_clusters = "\n".join(blocos_clusters)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Resultado da Clusterização Facial</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f4f5f7;
      color: #1f2937;
      margin: 0;
      padding: 24px;
    }}
    header {{
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    header h1 {{
      margin: 0 0 6px 0;
      font-size: 22px;
    }}
    header p {{
      margin: 0;
      color: #4b5563;
      font-size: 14px;
    }}
    .controles {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }}
    .controles button {{
      background: #f3f4f6;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      padding: 7px 12px;
      font-size: 13px;
      cursor: pointer;
      color: #1f2937;
    }}
    .controles button.ativo {{
      background: #2563eb;
      border-color: #2563eb;
      color: #ffffff;
      font-weight: 600;
    }}
    .controles label {{
      font-size: 13px;
      color: #374151;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      margin-left: 8px;
    }}
    .cluster {{
      background: #ffffff;
      border: 2px solid #cbd5e1;
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 24px;
    }}
    .cluster h2 {{
      margin: 0 0 14px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 18px;
    }}
    .cluster h2 small {{
      font-weight: normal;
      color: #6b7280;
      font-size: 14px;
    }}
    .grade {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
    }}
    .foto {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 8px;
    }}
    .par-imagens {{
      display: flex;
      align-items: flex-end;
      gap: 8px;
    }}
    .col-img {{
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .col-img small {{
      margin-top: 4px;
      font-size: 11px;
      color: #6b7280;
    }}
    .img-original img {{
      width: 112px;
      height: 136px;
      object-fit: cover;
      border-radius: 4px;
      display: block;
    }}
    .img-alinhada img {{
      width: 112px;
      height: 112px;
      border-radius: 4px;
      border: 1px solid #9ca3af;
      display: block;
      image-rendering: auto;
    }}
    .nome-arquivo {{
      margin-top: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #374151;
      word-break: break-all;
      text-align: center;
    }}
    body.modo-alinhadas .img-original {{
      display: none;
    }}
    body.modo-originais .img-alinhada {{
      display: none;
    }}
    body.ocultar-individuais .cluster-individual {{
      display: none;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Resultado da Clusterização de Faces</h1>
      <p>
        Total de faces: <strong>{total_faces}</strong> |
        Total de clusters: <strong>{total_grupos}</strong>
        (Clusters com 2+ fotos: <strong>{grupos_hdbscan}</strong> | Individuais: {grupos_individuais})
      </p>
    </div>
    <div class="controles">
      <button id="btn-ambas" class="ativo" onclick="definirModo('ambas')">Original + Alinhada (112x112)</button>
      <button id="btn-alinhadas" onclick="definirModo('alinhadas')">Apenas Alinhadas (112x112)</button>
      <button id="btn-originais" onclick="definirModo('originais')">Apenas Originais</button>
      <label>
        <input type="checkbox" id="chk-individuais" onchange="alternarIndividuais(this.checked)">
        Ocultar grupos individuais (1 foto)
      </label>
    </div>
  </header>
  {conteudo_clusters}
  <script>
    function definirModo(modo) {{
      document.body.classList.remove('modo-alinhadas', 'modo-originais');
      if (modo === 'alinhadas') document.body.classList.add('modo-alinhadas');
      if (modo === 'originais') document.body.classList.add('modo-originais');
      document.getElementById('btn-ambas').classList.toggle('ativo', modo === 'ambas');
      document.getElementById('btn-alinhadas').classList.toggle('ativo', modo === 'alinhadas');
      document.getElementById('btn-originais').classList.toggle('ativo', modo === 'originais');
    }}
    function alternarIndividuais(ocultar) {{
      document.body.classList.toggle('ocultar-individuais', ocultar);
    }}
  </script>
</body>
</html>
"""


def executar_etapa_resultado(abrir_navegador: bool = True) -> bool:
    """Etapa 4: Gera o HTML com as fotos originais e rostos alinhados 112x112 e abre no navegador."""
    print("\n=== Etapa 4: Visualização de Resultados ===")
    if not ARQUIVO_CLUSTERS.is_file():
        print(f"Arquivo de clusters não encontrado: {ARQUIVO_CLUSTERS}")
        print("Execute primeiro a Etapa 3 (STEP=clusterizar).")
        return False

    with ARQUIVO_CLUSTERS.open("r", encoding="utf-8") as arquivo:
        dados_clusters = json.load(arquivo)

    rostos_alinhados = None
    if ARQUIVO_ROSTOS_ALINHADOS.is_file():
        rostos_alinhados = np.load(ARQUIVO_ROSTOS_ALINHADOS, mmap_mode="r", allow_pickle=False)

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    conteudo_html = gerar_html_clusters(dados_clusters, rostos_alinhados)
    ARQUIVO_HTML_RESULTADO.write_text(conteudo_html, encoding="utf-8")

    print(f"Relatório HTML gerado (com visualização 112x112): {ARQUIVO_HTML_RESULTADO}")
    if abrir_navegador:
        webbrowser.open(ARQUIVO_HTML_RESULTADO.resolve().as_uri())
    return True
