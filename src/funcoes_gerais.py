"""
Módulo de Funções Gerais e Utilitárias.

Este módulo contém funções auxiliares para manipulação de texto, contagem de tokens
e pré-processamento de dados que são reutilizadas em diversas partes do sistema.
"""

import logger
from logger import LogLevel
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import tiktoken

# Cache do tokenizador no nível do módulo (evita re-instanciação a cada chamada)
_ENCODER_CACHE = tiktoken.get_encoding("o200k_harmony")


def calcular_tokens(texto: str, modelo: str = "gpt-oss") -> int:
    """
    Calcula a estimativa de tokens de um texto para um modelo específico.
    Utiliza a biblioteca tiktoken para tokenização BPE.

    Args:
        texto (str): O texto a ser tokenizado.
        modelo (str): Identificador do modelo ('gpt-oss' usa o tokenizador o200k_harmony).

    Returns:
        int: Número de tokens no texto. Retorna -1 se o modelo não for suportado.
    """
    try:
        tokens = _ENCODER_CACHE.encode(str(texto))
        return len(tokens)
    except Exception as e:
        logger.log(f"Erro ao calcular tokens: {e}", level=LogLevel.ERROR)
    return -1


def pre_processamento_geral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a higienização e normalização estrutural do DataFrame de prontuários.
    Foca na coluna 'descricao', removendo quebras de linha excessivas e caracteres
    de controle que podem confundir o processamento do LLM ou Regex.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'descricao'.

    Returns:
        pd.DataFrame: DataFrame com o texto limpo e normalizado.
    """

    def _limpar_texto(texto: str) -> str:
        """Helper para limpar uma única string."""
        if pd.isna(texto):
            return ""

        t = str(texto)

        # 1. Normaliza quebras de linha (\r\n -> \n)
        t = t.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Remove espaços em branco (espaço, tab) imediatamente antes ou depois de \n
        # Isso evita quebras de linha "fantasmas" que quebram o fluxo da frase.
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n[ \t]+", "\n", t)

        # 3. Colapsa múltiplas quebras de linha (3 ou mais viram 2)
        # Mantém separação de parágrafos mas remove grandes espaços vazios.
        t = re.sub(r"\n{3,}", "\n\n", t)

        # NOTA: Colchetes NÃO são mais substituídos.
        # O sistema usa TAGs dinâmicas (caracteres únicos) para evitar conflitos.

        return t.strip()

    logger.log("Iniciando pré-processamento estrutural do DataFrame...", level=LogLevel.NORMAL)
    df["descricao"] = df["descricao"].apply(_limpar_texto)
    logger.log("Limpeza de formatação concluída.", level=LogLevel.NORMAL)

    return df


def identificar_caracteres_mascara(texto_completo: str, quantidade: int = 2) -> tuple:
    """
    Identifica N caracteres que NÃO existem no texto para uso como máscara/TAGs.

    Uso:
        - TAGs (2 chars: ini/fim): identificar_caracteres_mascara(texto, 2)
        - Máscara (2 chars: pii/unknown): identificar_caracteres_mascara(texto, 2)

    Args:
        texto_completo: Corpus completo concatenado para análise.
        quantidade: Número de caracteres únicos a identificar.

    Returns:
        Tupla com os caracteres selecionados.
    """
    # Candidatos em ordem de preferência:
    # 1. Símbolos Unicode raros (ideais para TAGs - nunca aparecem em prontuários)
    # 2. Símbolos comuns ASCII (para máscaras - mais visíveis em relatórios)
    candidatos = [
        "⦃",
        "⦄",  # Curly com ponto (U+2983, U+2984)
        "⟦",
        "⟧",  # Colchetes matemáticos (U+27E6, U+27E7)
        "⟨",
        "⟩",  # Angle brackets (U+27E8, U+27E9)
        "❮",
        "❯",  # Heavy angle (U+276E, U+276F)
        "*",
        "#",
        "@",
        "&",
        "%",
        "§",
        "¶",
        "†",
        "‡",
        "Ψ",
        "Ω",
        "β",
        "Σ",
    ]

    selecionados = []
    caracteres_texto = set(texto_completo)

    for c in candidatos:
        if c not in caracteres_texto:
            selecionados.append(c)
            if len(selecionados) == quantidade:
                return tuple(selecionados)

    raise ValueError(f"Não foi possível encontrar {quantidade} caracteres seguros no dataset.")


# Uso:
# python -c "import funcoes_gerais as f
# f.verificar_alinhamento_arquivos('saida/<timestamp>_original_<sessao>_<modelo>.txt', 'saida/<timestamp>_mascara_<sessao>_<modelo>.mask')"
def verificar_alinhamento_arquivos(original, mascarado):
    """
    Verifica o alinhamento do dataset normal e do mascarado.
    Retorna a quantidade de bytes desalinhados
    """
    with open(mascarado + ".meta", encoding="utf-8") as f:
        dados = json.load(f)

    char_pii = dados["char_pii"]
    char_unk = dados["char_unk"]

    contador = 0
    for n, i in enumerate(zip(open(original, encoding="utf-8").read(), open(mascarado, encoding="utf-8").read())):
        if i[1] in (i[0], char_pii, char_unk):
            continue
        contador += 1

    logger.log(f"\n{contador} bytes desalinhados.\n", level=LogLevel.NORMAL)

    return contador


def carregar_contador(mapa: dict[str, int]) -> int:
    """Calcula o próximo ID disponível baseado no maior ID existente no mapa."""
    return 1 if not mapa else max(mapa.values()) + 1


def gerar_combinacoes(ids: list[int]) -> list[tuple[int, ...]]:
    """
    Gera todas as combinações não-vazias (power set) de uma lista de IDs.
    Ex: [63, 65, 72] → [(63,), (65,), (72,), (63,65), (63,72), (65,72), (63,65,72)]
    """
    combos = []
    for r in range(1, len(ids) + 1):
        combos.extend(combinations(ids, r))
    return combos


def _merge_mapas(lista_mapas: list[dict]) -> dict:
    """
    Une múltiplos mapas de conhecimento, reindexando IDs sequencialmente
    para evitar colisões entre sessões diferentes.
    """
    merged = {}
    for mapa in lista_mapas:
        for key in mapa:
            if key not in merged:
                merged[key] = len(merged) + 1
    return merged


def _intersect_mapas(lista_mapas: list[dict]) -> dict:
    """
    Intersecta múltiplos mapas de conhecimento por chave textual.
    Retém apenas termos presentes em TODOS os mapas, reindexando IDs
    sequencialmente. Retorna {} se qualquer mapa for vazio ou lista vazia.
    """
    if not lista_mapas:
        return {}
    chaves_comuns = set(lista_mapas[0].keys())
    for mapa in lista_mapas[1:]:
        chaves_comuns &= set(mapa.keys())
    resultado = {}
    for chave in sorted(chaves_comuns):  # sorted para determinismo nos testes
        resultado[chave] = len(resultado) + 1
    return resultado


def sanitizar_nome_sessao(nome: str) -> str:
    """
    Sanitiza nome_identificador para uso em nomes de arquivo.
    Ex: 'alfa 6 [qwen3.5:4b]' → 'alfa-6-qwen3.5-4b'
    """
    result = nome.replace("[", "-").replace("]", "").replace(":", "-").replace(" ", "-")
    result = re.sub(r"-{2,}", "-", result)  # Colapsa dashes consecutivos
    return result.strip("-")


def sanitizar_lista_tags(lista_termos: list[str], texto_contexto: str) -> list[str]:
    """
    Remove itens da lista que sejam puramente 'TAG NÚMERO' (ex: 'PERSON 311'),
    MAS apenas se essa tag realmente existir no texto de entrada como uma tag de sistema [ PERSON 311 ].

    Isso evita falsos positivos (apagar texto real que parece tag) e remove alucinações (tag vazada).
    """
    if not lista_termos:
        return []

    lista_limpa = []
    # Regex para identificar formato de tag: PALAVRA + DIGITOS
    padrao_tag = re.compile(
        r"^(BOILERPLATE|PERSON|LOCATION|INFO|DATE|EMAIL|PHONE|UNKNOWN|REVISAO|CPF|CNS)\s*\d+$", re.IGNORECASE
    )

    for t in lista_termos:
        if not isinstance(t, str):
            continue
        termo_limpo = t.strip()
        if padrao_tag.match(termo_limpo):
            # O termo parece uma tag. Vamos ver se ele é UMA TAG no texto original.
            # Busca por [ TERMO ] com flexibilidade de espaços
            # Ex: Se termo é "PERSON 311", busca "\[\s*PERSON\s*311\s*\]"
            termo_regex = re.escape(termo_limpo).replace(r"\ ", r"\s*")
            padrao_no_texto = re.compile(rf"\[\s*{termo_regex}\s*\]", re.IGNORECASE)

            if padrao_no_texto.search(texto_contexto):
                # É uma tag do sistema que vazou sem colchetes -> REMOVE
                continue
            else:
                # Parece tag, mas não está entre colchetes no texto.
                # Pode ser um dado real (ex: "Código EMAIL 20"). -> MANTÉM
                lista_limpa.append(t)
        else:
            lista_limpa.append(t)

    return lista_limpa


# =============================================
# Plotagem de Métricas de LLM (Sessões)
# =============================================


def _parsear_modelo_de_nome_arquivo(nome_arquivo: str) -> str:
    """Extrai os nomes concatenados dos modelos a partir do nome do arquivo JSON."""
    nome_base = nome_arquivo.replace(".json", "")
    partes = nome_base.split("_")
    # A estrutura esperada é: DATA_HORA_mascara_IDS_MODELOS
    # O modelo começa no 4º underscore.
    if len(partes) >= 5:
        return "_".join(partes[4:])
    return nome_base


def _parsear_modo_de_nome_arquivo(nome_arquivo: str) -> str:
    """Extrai 'Uniao', 'Interseccao' ou 'Unico' do prefixo do label no nome do arquivo JSON.

    Estrutura esperada: DATA_HORA_mascara_IDS_MODO_NOMES.json
    Onde MODO é 'uniao', 'interseccao' ou 'unico'.
    """
    nome_base = nome_arquivo.replace(".json", "")
    partes = nome_base.split("_")
    if len(partes) >= 5:
        modo_raw = partes[4].lower()
        if modo_raw == "interseccao":
            return "Interseccao"
        elif modo_raw == "unico":
            return "Unico"
    return "Uniao"  # default (uniao ou arquivo sem prefixo esperado)


def _parsear_nivel_de_nome_arquivo(nome_arquivo: str) -> int:
    """Extrai o número de sessões combinadas (nível K) do nome do arquivo JSON.

    IDs seguem o padrão 's{ID1}-s{ID2}-...', ex: 's57-s58-s59' = k=3.
    """
    nome_base = nome_arquivo.replace(".json", "")
    partes = nome_base.split("_")
    if len(partes) >= 4:
        ids_str = partes[3]  # ex: s57-s58-s59
        return len(ids_str.split("-s"))
    return 1


def _agregar_metricas_json(diretorio: str) -> pd.DataFrame:
    dados = []
    caminho = Path(diretorio)

    for arquivo in caminho.glob("*.json"):
        try:
            with open(arquivo, encoding="utf-8") as f:
                json_data = json.load(f)

            metricas = json_data.get("metricas", {})
            if not metricas:
                continue

            modelo = _parsear_modelo_de_nome_arquivo(arquivo.name)
            modo = _parsear_modo_de_nome_arquivo(arquivo.name)
            nivel = _parsear_nivel_de_nome_arquivo(arquivo.name)

            dados.append(
                {
                    "Modelo": modelo,
                    "Modo": modo,
                    "Nivel": nivel,
                    "Precision": metricas.get("precision", 0.0),
                    "Recall": metricas.get("recall", 0.0),
                    "F1": metricas.get("f1_score", 0.0),
                    "F2_Score": metricas.get("f2_score", 0.0),
                }
            )
        except Exception as e:
            logger.log(f"Erro ao ler {arquivo.name}: {e}", level=LogLevel.ERROR)

    df = pd.DataFrame(dados)
    if not df.empty:
        df = df.sort_values(by=["F2_Score", "Precision"], ascending=[False, False])

    return df


def plotar_metricas_relatorios(diretorio_origem: str, caminho_salvar_png: str, top_n: int = 20):
    """
    Gera duas visualizações estatísticas dos relatórios JSON:
    1. Gráfico de Barras horizontais com os Top N combos por F2-Score, coloridos por Modo.
    2. Gráfico Scatter global Precision vs Recall, colorido por Modo, marker por Nível K.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    df = _agregar_metricas_json(diretorio_origem)
    if df.empty:
        logger.log("Nenhum dado encontrado para plotar.", level=LogLevel.NORMAL)
        return
        
    _desenhar_grafico_matplotlib(df, caminho_salvar_png, top_n)

def plotar_metricas_via_banco(supabase_client, caminho_salvar_png: str, top_n: int = 20):
    import pandas as pd
    registros = supabase_client.buscar_todas_as_combinacoes_validas()
    if not registros:
        logger.log("Nenhum dado encontrado no BD para plotar.", level=LogLevel.NORMAL)
        return

    dados = []
    for r in registros:
        nome_completo = r.get("label_combo", "")
        modo_bruto = nome_completo.split("_")[0] if "_" in nome_completo else "uniao"

        dados.append({
            "Label": r.get("label_combo", ""),
            "Modelo": r.get("label_combo", ""),
            "Grupo": r.get("grupo", "default"),
            "Modo": modo_bruto.capitalize(),
            "Nivel": r.get("nivel_k", 1),
            "Precision": r.get("precision", 0.0) or 0.0,
            "Recall": r.get("recall", 0.0) or 0.0,
            "F1": r.get("f1_score", 0.0) or 0.0,
            "F2_Score": r.get("f2_score", 0.0) or 0.0,
        })

    df = pd.DataFrame(dados)
    if df.empty:
        return
        
    df = df.sort_values(by=["F2_Score", "Precision"], ascending=[False, False])
    _desenhar_grafico_matplotlib(df, caminho_salvar_png, top_n)

def _desenhar_grafico_matplotlib(df, caminho_salvar_png: str, top_n: int):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Paleta e marcadores por Modo
    paleta_modo = {"Uniao": "#4C9BE8", "Interseccao": "#E8834C", "Unico": "#8E4CE8"}

    # Coluna legível para nível K
    df["Nivel_Label"] = df["Nivel"].apply(lambda k: f"k={k}")
    marcadores_nivel = {label: m for label, m in zip(
        sorted(df["Nivel_Label"].unique()), ["o", "s", "^", "D", "v", "P", "*"]
    )}

    n_bars = min(top_n, len(df))
    df_top = df.head(n_bars).copy()

    # Trunca labels longos para caber no eixo Y
    _MAX_LABEL = 48
    df_top["Label"] = df_top["Modelo"].apply(
        lambda x: (x[:_MAX_LABEL - 3] + "…") if len(x) > _MAX_LABEL else x
    )

    # Altura dinâmica: mínimo 10, cresce com o número de barras (0.7 por barra + margens)
    altura = max(10, n_bars * 0.7 + 5)
    fig = plt.figure(figsize=(20, altura), dpi=120)
    fig.suptitle("Métricas de Anonimização — União vs Intersecção", fontsize=13, fontweight="bold", y=1.005)

    # GridSpec: barra ocupa 60%, scatter 40%
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.35, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # --- Gráfico 1: Barras do Pódio (Top N F2-Score, coloridas por Modo) ---
    cores_top = [paleta_modo.get(m, "#AAAAAA") for m in df_top["Modo"]]
    ax1.barh(df_top["Label"], df_top["F2_Score"], color=cores_top, edgecolor="white", linewidth=0.5)
    ax1.set_title(f"Top {n_bars} Combinações por F2-Score", fontsize=11, pad=10)
    ax1.set_xlim(0, 1.22)
    ax1.set_xlabel("F2-Score", fontsize=9)
    ax1.set_ylabel("")
    ax1.tick_params(axis="y", labelsize=7.5)
    ax1.invert_yaxis()  # melhor combo no topo
    ax1.set_ylim(n_bars - 0.5, -0.5)  # força limites explícitos para não cortar barras
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Labels nas barras (compactos)
    for i, (_, row) in enumerate(df_top.iterrows()):
        ax1.text(
            min(row["F2_Score"] + 0.008, 1.03), i,
            f"F2:{row['F2_Score']:.3f}  R:{row['Recall']:.3f}  P:{row['Precision']:.3f}  [{row['Modo'][0].upper()}k{row['Nivel']}]",
            va="center", fontsize=6.5, color="#333333", clip_on=True,
        )

    # Legenda de modo
    from matplotlib.patches import Patch
    legenda_modo = [Patch(facecolor=c, label=m, edgecolor="white") for m, c in paleta_modo.items()]
    ax1.legend(handles=legenda_modo, title="Modo", loc="lower right", fontsize=8, title_fontsize=8)

    # --- Gráfico 2: Scatter Global (Modo=cor, Nível=marcador) ---
    for nivel_label, grupo in df.groupby("Nivel_Label"):
        for modo, subgrupo in grupo.groupby("Modo"):
            ax2.scatter(
                subgrupo["Precision"],
                subgrupo["Recall"],
                c=paleta_modo.get(modo, "#AAAAAA"),
                marker=marcadores_nivel.get(nivel_label, "o"),
                s=subgrupo["F2_Score"] * 180 + 20,
                alpha=0.72,
                label=f"{modo} {nivel_label}",
                edgecolors="white",
                linewidths=0.4,
            )

    prec_min = df["Precision"].min()
    rec_min = df["Recall"].min()
    ax2.set_title(f"Precision × Recall global ({len(df)} combos)", fontsize=11, pad=10)
    ax2.set_xlim(max(0, prec_min * 0.97), 1.03)
    ax2.set_ylim(max(0, rec_min * 0.97), 1.03)
    ax2.set_xlabel("Precision", fontsize=9)
    ax2.set_ylabel("Recall", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(title="Modo / Nível", fontsize=7.5, title_fontsize=8, loc="lower left",
               markerscale=1.2, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(caminho_salvar_png, dpi=150, bbox_inches="tight")
    plt.close()
    logger.log(f"Gráfico salvo com sucesso em: {caminho_salvar_png}", level=LogLevel.NORMAL)

