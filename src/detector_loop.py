"""
Módulo de Detecção de Loops de Raciocínio (Reasoning Loop).

Detecta repetições cíclicas em texto recebido via streaming de LLMs.
Inspirado na técnica DeRep (Liu et al., 2025 — arXiv:2504.12608),
adaptada para detecção em tempo real durante geração de texto.

Quatro detectores em cascata (mais barato primeiro):
1. Janela exata (Det-1) — O(janela × limiar), a cada chunk
2. rep-n ratio (Det-2) — O(n_tokens), a cada N chars
3. Cosine similarity por linhas (Det-3) — O(janela²), a cada nova linha
4. Blocos de linhas repetidos (Det-4) — O(N × salto), a cada nova linha
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DetectorLoop:
    """
    Detecta repetições cíclicas em texto recebido via streaming.

    Uso típico:
        detector = DetectorLoop(config)
        for chunk in stream:
            resultado = detector.alimentar(chunk)
            if resultado:
                # Loop detectado: resultado["tipo"], resultado["detalhes"]
                break
    """

    # Defaults para todos os parâmetros
    _DEFAULTS = {
        "janela_chars": 100,
        "limiar_janelas": 3,
        "rep_n_tamanho": 3,
        "rep_n_threshold": 0.5,
        "rep_n_min_tokens": 150,
        "rep_n_intervalo": 500,
        "rep_n_janela_tokens": 1000,
        "cosine_janela_linhas": 10,
        "cosine_threshold": 0.85,
        "cosine_min_linhas": 20,
        "cosine_ratio_pares": 0.6,
        "bloco_limiar": 3,
        "bloco_min_linhas": 6,
        "bloco_min_chars_linha": 10,
    }

    def __init__(self, config: dict | None = None):
        """
        Inicializa o detector com parâmetros configuráveis.

        Args:
            config: Dict com parâmetros. Chaves ausentes usam os defaults.
                    Parâmetros suportados:
                        janela_chars, limiar_janelas,
                        rep_n_tamanho, rep_n_threshold, rep_n_min_tokens,
                        rep_n_intervalo, rep_n_janela_tokens,
                        cosine_janela_linhas, cosine_threshold, cosine_min_linhas, cosine_ratio_pares,
                        bloco_limiar, bloco_min_linhas, bloco_min_chars_linha,
                        detectores_ativos (None=todos, ou set ex: {1,2,3} ou {4})
        """
        cfg = {**self._DEFAULTS, **(config or {})}

        # Det-1: Janela exata
        self._janela_chars = cfg["janela_chars"]
        self._limiar_janelas = cfg["limiar_janelas"]

        # Det-2: rep-n ratio
        self._rep_n_tamanho = cfg["rep_n_tamanho"]
        self._rep_n_threshold = cfg["rep_n_threshold"]
        self._rep_n_min_tokens = cfg["rep_n_min_tokens"]
        self._rep_n_intervalo = cfg["rep_n_intervalo"]
        self._rep_n_janela_tokens = cfg["rep_n_janela_tokens"]

        # Det-3: Cosine similarity
        self._cosine_janela_linhas = cfg["cosine_janela_linhas"]
        self._cosine_threshold = cfg["cosine_threshold"]
        self._cosine_min_linhas = cfg["cosine_min_linhas"]
        self._cosine_ratio_pares = cfg["cosine_ratio_pares"]

        # Det-4: Blocos de linhas repetidos
        self._bloco_limiar = cfg["bloco_limiar"]
        self._bloco_min_linhas = cfg["bloco_min_linhas"]
        self._bloco_min_chars_linha = cfg["bloco_min_chars_linha"]

        # Seleção de detectores ativos (None = todos)
        self._detectores_ativos: set | None = cfg.get("detectores_ativos")

        # Estado interno
        self._buffer = ""
        self._posicao_inicio_loop: int | None = None
        self._chars_desde_ultimo_rep_n = 0
        self._linhas_contadas = 0

        # Estado Det-4
        self._linhas_det4: list[tuple[str, int]] = []  # (texto_stripped, pos_char)
        self._indice_linhas_det4: dict[int, list[int]] = {}  # hash(texto) -> [idx]
        self._linhas_sig_det4 = 0  # Contador de linhas significativas (warmup)
        self._parcial_det4 = ""  # Buffer da linha sendo montada token a token
        self._pos_inicio_linha_det4 = 0  # Pos em chars do inicio da linha parcial

    def alimentar(self, texto: str) -> dict | None:
        """
        Recebe um chunk de texto (streaming) e verifica se há loop.

        Args:
            texto: Chunk de texto recebido do streaming do LLM.

        Returns:
            None se não detectou loop.
            Dict com {"tipo": str, "detalhes": str} se detectou.
        """
        self._buffer += texto
        self._chars_desde_ultimo_rep_n += len(texto)
        ativos = self._detectores_ativos  # None = todos

        # 1. Det-1: Janela exata
        if ativos is None or 1 in ativos:
            resultado_janela = self._verificar_janela_exata()
            if resultado_janela:
                return resultado_janela

        # 2. Det-2: rep-n ratio (a cada rep_n_intervalo chars)
        if ativos is None or 2 in ativos:
            if self._chars_desde_ultimo_rep_n >= self._rep_n_intervalo:
                self._chars_desde_ultimo_rep_n = 0
                resultado_rep_n = self._verificar_rep_n()
                if resultado_rep_n:
                    return resultado_rep_n

        # 3. Det-3: Cosine similarity (a cada nova linha)
        if ativos is None or 3 in ativos:
            linhas_atuais = self._buffer.count("\n")
            if linhas_atuais > self._linhas_contadas:
                self._linhas_contadas = linhas_atuais
                resultado_cosine = self._verificar_cosine_linhas()
                if resultado_cosine:
                    return resultado_cosine

        # 4. Det-4: Blocos de linhas repetidos (a cada nova linha completa)
        if ativos is None or 4 in ativos:
            self._parcial_det4 += texto
            while "\n" in self._parcial_det4:
                pos_nl = self._parcial_det4.index("\n")
                linha_completa = self._parcial_det4[:pos_nl].strip()
                pos_char = self._pos_inicio_linha_det4

                # Registra TODAS as linhas (inclusive vazias) para comparacao exata
                idx = len(self._linhas_det4)
                self._linhas_det4.append((linha_completa, pos_char))
                h = hash(linha_completa)
                self._indice_linhas_det4.setdefault(h, []).append(idx)

                if len(linha_completa) >= self._bloco_min_chars_linha:
                    self._linhas_sig_det4 += 1

                # Busca blocos apenas se warmup atingido e linha eh significativa
                if (self._linhas_sig_det4 >= self._bloco_min_linhas
                        and len(linha_completa) >= self._bloco_min_chars_linha):
                    resultado_bloco = self._verificar_blocos_linhas()
                    if resultado_bloco:
                        return resultado_bloco

                # Avanca posicao para a proxima linha
                self._pos_inicio_linha_det4 = pos_char + pos_nl + 1
                self._parcial_det4 = self._parcial_det4[pos_nl + 1:]

        return None

    def obter_posicao_inicio_loop(self) -> int | None:
        """
        Retorna a posição (offset em chars) no buffer onde o loop começou.

        Usado pelo Mit-2 para truncar o raciocínio antes da injeção.
        Retorna None se nenhum loop foi detectado.
        """
        return self._posicao_inicio_loop

    def reset(self):
        """Limpa todos os buffers e estado internos."""
        self._buffer = ""
        self._posicao_inicio_loop = None
        self._chars_desde_ultimo_rep_n = 0
        self._linhas_contadas = 0
        # Det-4
        self._linhas_det4 = []
        self._indice_linhas_det4 = {}
        self._linhas_sig_det4 = 0
        self._parcial_det4 = ""
        self._pos_inicio_linha_det4 = 0

    # ============================================================
    # Detectores internos
    # ============================================================

    def _verificar_janela_exata(self) -> dict | None:
        """
        Det-1: Compara as últimas N janelas de K chars por match exato.

        Detecta padrões AAA (período 1): mesma linha repetida.
        Lógica extraída de ClienteOllama.executar_prompt() L158-190.
        """
        tamanho_necessario = self._janela_chars * self._limiar_janelas
        if len(self._buffer) < tamanho_necessario:
            return None

        janela_atual = self._buffer[-self._janela_chars :]
        em_loop = True
        for j in range(2, self._limiar_janelas + 1):
            inicio = -self._janela_chars * j
            fim = -self._janela_chars * (j - 1)
            janela_anterior = self._buffer[inicio:fim]
            if janela_atual != janela_anterior:
                em_loop = False
                break

        if em_loop:
            # Posição de início: antes das janelas repetidas
            self._posicao_inicio_loop = len(self._buffer) - tamanho_necessario
            return {
                "tipo": "janela_exata",
                "detalhes": (
                    f"Det-1: {self._limiar_janelas} janelas de {self._janela_chars} chars idênticas"
                ),
            }

        return None

    def _verificar_rep_n(self) -> dict | None:
        """
        Det-2: Calcula razão de n-grams repetidos (métrica rep-n do DeRep).

        rep-n = 1 - |unique_ngrams| / |total_ngrams|
        Valores altos indicam repetição. Captura AAA, ABAB, ciclos.
        """
        tokens = self._buffer.split()
        if len(tokens) < self._rep_n_min_tokens:
            return None

        # Avalia os últimos N tokens (configurável para cobrir ciclos longos)
        tokens_recentes = tokens[-self._rep_n_janela_tokens:]
        n = self._rep_n_tamanho
        if len(tokens_recentes) <= n:
            return None

        ngrams = [tuple(tokens_recentes[i : i + n]) for i in range(len(tokens_recentes) - n + 1)]
        total = len(ngrams)
        unicos = len(set(ngrams))
        ratio = 1 - unicos / max(total, 1)

        if ratio > self._rep_n_threshold:
            # Estima posição via cosine deslizante (mais preciso para listas formatadas)
            # Se cosine falhar, cai no fallback de rep_n reverso
            posicao = self._estimar_posicao_cosine_deslizante()
            if posicao is None:
                posicao = self._estimar_posicao_inicio_rep_n(tokens)
            self._posicao_inicio_loop = posicao
            return {
                "tipo": "rep_n",
                "detalhes": (
                    f"Det-2: rep-{n} ratio = {ratio:.3f} (threshold: {self._rep_n_threshold})"
                ),
            }

        return None

    def _estimar_posicao_cosine_deslizante(self) -> int | None:
        """
        Estima posição de início do loop usando cosine deslizante de trás para frente.

        Compara blocos de N linhas consecutivas. Quando a similaridade média
        cai abaixo do threshold, encontrou a fronteira do loop.
        Mais preciso que rep_n para distinguir listas formatadas de loops reais.
        """
        linhas = [l.strip() for l in self._buffer.split("\n") if l.strip()]
        janela = self._cosine_janela_linhas
        if len(linhas) < janela * 2:
            return None

        # Busca de trás para frente: compara janela_atual com janela_anterior
        linhas_buffer = self._buffer.split("\n")
        for idx in range(len(linhas) - janela, janela - 1, -1):
            bloco_atual = linhas[idx : idx + janela]
            bloco_anterior = linhas[idx - janela : idx]

            try:
                todos = bloco_anterior + bloco_atual
                tfidf = TfidfVectorizer().fit_transform(todos)
                sim = cosine_similarity(tfidf[:janela], tfidf[janela:])
                # Similaridade média entre os blocos
                media_sim = sim.diagonal().mean()
            except ValueError:
                continue

            if media_sim < self._cosine_threshold:
                # Fronteira encontrada: o loop começa em idx
                # Converter índice de linha para posição em chars
                pos = 0
                linhas_nao_vazias = 0
                for linha in linhas_buffer:
                    if linha.strip():
                        if linhas_nao_vazias >= idx:
                            return pos
                        linhas_nao_vazias += 1
                    pos += len(linha) + 1
                return pos

        return None

    def _estimar_posicao_inicio_rep_n(self, tokens: list[str]) -> int:
        """
        Fallback: estima posição via rep_n de trás para frente.

        Usado quando o cosine deslizante não consegue encontrar a fronteira.
        """
        n = self._rep_n_tamanho
        janela_tokens = 100
        passo = 25
        threshold_posicao = max(self._rep_n_threshold + 0.2, 0.7)

        posicoes = list(range(0, max(len(tokens) - janela_tokens, 1), passo))
        posicoes.reverse()

        ultima_posicao_limpa = len(tokens)
        for i in posicoes:
            trecho = tokens[i : i + janela_tokens]
            if len(trecho) <= n:
                continue
            ngrams = [tuple(trecho[j : j + n]) for j in range(len(trecho) - n + 1)]
            total = len(ngrams)
            unicos = len(set(ngrams))
            ratio = 1 - unicos / max(total, 1)
            if ratio <= threshold_posicao:
                ultima_posicao_limpa = i + janela_tokens
                break

        if ultima_posicao_limpa >= len(tokens):
            return len(self._buffer) // 2
        return len(" ".join(tokens[:ultima_posicao_limpa]))

    def _verificar_cosine_linhas(self) -> dict | None:
        """
        Det-3: Similaridade de cosseno entre as últimas N linhas via TF-IDF.

        Usa TfidfVectorizer + cosine_similarity do sklearn.
        Captura ABAB (loop3), ciclos com variação textual (loop4).
        """
        linhas = [l.strip() for l in self._buffer.split("\n") if l.strip()]
        if len(linhas) < self._cosine_min_linhas:
            return None

        recentes = linhas[-self._cosine_janela_linhas :]
        n = len(recentes)
        if n < 2:
            return None

        try:
            tfidf = TfidfVectorizer().fit_transform(recentes)
            sim_matrix = cosine_similarity(tfidf)
        except ValueError:
            # Pode ocorrer se todas as linhas forem vazias ou stop words
            return None

        pares_similares = 0
        total_pares = n * (n - 1) / 2
        if total_pares == 0:
            return None

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > self._cosine_threshold:
                    pares_similares += 1

        ratio_pares = pares_similares / total_pares

        if ratio_pares > self._cosine_ratio_pares:
            # Estimar posição: a partir das últimas cosine_janela_linhas
            posicao = self._estimar_posicao_inicio_cosine(linhas)
            self._posicao_inicio_loop = posicao
            return {
                "tipo": "cosine_sim",
                "detalhes": (
                    f"Det-3: {pares_similares}/{int(total_pares)} pares similares "
                    f"(ratio: {ratio_pares:.3f}, threshold: {self._cosine_ratio_pares})"
                ),
            }

        return None

    def _estimar_posicao_inicio_cosine(self, linhas: list[str]) -> int:
        """
        Estima a posição no buffer onde a repetição por cosine começa.

        Busca o início do bloco de linhas repetitivas.
        """
        # Posição das últimas cosine_janela_linhas linhas no buffer
        todas_linhas = [l for l in self._buffer.split("\n") if l.strip()]
        n_total = len(todas_linhas)
        n_janela = self._cosine_janela_linhas
        idx_inicio_janela = max(0, n_total - n_janela)

        # Encontra a posição em chars do início dessa janela
        pos = 0
        linhas_buffer = self._buffer.split("\n")
        linhas_nao_vazias = 0
        for linha in linhas_buffer:
            if linha.strip():
                if linhas_nao_vazias >= idx_inicio_janela:
                    return pos
                linhas_nao_vazias += 1
            pos += len(linha) + 1  # +1 para \n

        return len(self._buffer) // 2

    def _verificar_blocos_linhas(self) -> dict | None:
        """
        Det-4: Detecta blocos de linhas que se repetem N vezes (match exato com strip).

        Busca para trás via hash a partir da linha atual, verifica se existem
        N blocos idênticos espaçados por 'salto' linhas. Compara TODAS as linhas
        internas ao bloco (inclusive vazias), garantindo detecção exata.

        Salto mínimo = 2 (blocos de 1 linha são cobertos pelo Det-1).
        """
        linhas = self._linhas_det4
        n = len(linhas)
        N = self._bloco_limiar
        idx = n - 1
        max_salto = idx // N

        if max_salto < 2:
            return None

        # Busca via hash: posicoes anteriores com mesma linha (O(1) lookup)
        linha_atual = linhas[idx][0]
        h = hash(linha_atual)
        candidatos = self._indice_linhas_det4.get(h, [])

        # Iterar do mais recente ao mais antigo (saltos menores primeiro)
        for c in range(len(candidatos) - 1, -1, -1):
            pos_match = candidatos[c]
            if pos_match >= idx:
                continue
            salto = idx - pos_match
            if salto < 2:
                continue
            if salto > max_salto:
                break  # candidatos anteriores terao salto ainda maior

            # Verificar que cabem N blocos completos
            primeiro_idx = idx - (N - 1) * salto
            if primeiro_idx < 0:
                continue

            # Verificar TODAS as linhas do bloco em TODAS as N posicoes
            confirmado = True
            for offset in range(salto):
                ref = linhas[idx - offset][0]
                for k in range(1, N):
                    if linhas[idx - offset - k * salto][0] != ref:
                        confirmado = False
                        break
                if not confirmado:
                    break

            if confirmado:
                inicio_segundo_bloco = primeiro_idx + salto
                self._posicao_inicio_loop = linhas[inicio_segundo_bloco][1]
                return {
                    "tipo": "bloco_linhas",
                    "detalhes": (
                        f"Det-4: período={salto} linhas × {N} blocos"
                    ),
                }

        return None
