"""
Módulo de Lógica de Anonimização.

Implementa a classe central `Anonimizacao`, responsável por gerenciar o ciclo de vida
da ocultação de dados. O sistema opera de forma híbrida e incremental:

1.  **Regex**: Identifica padrões óbvios (CPF, Datas, CNS) instantaneamente.
2.  **Memória Associativa**: Armazena termos já identificados (ex: "Dr. House") e os
    substitui automaticamente em ocorrências futuras sem consultar o LLM.
3.  **Boilerplates**: Identifica frases médicas comuns e repetitivas (ex: "Paciente em bom estado geral")
    para reduzi-las a tags, economizando tokens e focando a atenção do LLM no que é único (e potencialmente sensível).
4.  **Integração LLM**: Prepara e limpa o texto antes de enviar para a IA, e processa as respostas JSON.
"""

import logger
from logger import LogLevel
import json
import re
from collections import Counter

import pandas as pd

from a01_platform import config_router as config


class Anonimizacao:
    """
    Motor de anonimização híbrido.
    Combina Expressões Regulares (Regex) e aprendizado incremental (Dicionários)
    para anonimizar dados sensíveis antes de enviar ao LLM.
    """

    # --- CONSTANTES DE TIPOS DE TAGS ---
    # NOTA: TAG_INI e TAG_FIM agora são dinâmicos, definidos no __init__

    # Tags de Identificação (utilizados dentro do texto e no banco)
    TAG_NOME = "PERSON"
    TAG_END = "LOCATION"
    TAG_OUTROS = "INFO"
    TAG_BOILERPLATE = "BOILERPLATE"
    TAG_REVISAO = "REVIEW"
    TAG_CPF = "CPF"
    TAG_PHONE = "PHONE"
    TAG_EMAIL = "EMAIL"
    TAG_DATE = "DATE"
    TAG_UNKNOWN = "UNKNOWN"

    # Constantes de configuração
    MIN_CHARS_SAFE_PHRASE = 15  # Tamanho mínimo de caracteres para considerar fragmento como boilerplate
    MIN_WORDS_SAFE_PHRASE = 3  # Mínimo de palavras para evitar fragmentos espúrios com espaços

    # Delimitador interno para separar segmentos em processamento de N-Grams e PIIs
    # Construído dinamicamente no __init__ para evitar colisão com o corpus
    DELIMITADOR_INTERNO: str = "[[["  # Default fallback; será sobrescrito no __init__

    # Regex pré-compiladas para alta performance em padrões rígidos
    REGEX_PADROES = {
        "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}[-.]\d{2}\b"),
        # Valida dia (01-31), mês (01-12) e ano (2 ou 4 dígitos)
        "DATA": re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/\d{2,4}\b"),
        # (?i) torna case-insensitive. Captura dia + "de" + mês (completo ou abrev) + opcional ano
        "DATA_EXTENSO": re.compile(
            r"(?i)\b\d{1,2}\s+de\s+(?:jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)(?:\s+de\s+\d{2,4})?\b"
        ),
        # Telefone: Exige DDD com parênteses ou separador para evitar falsos positivos
        "TEL": re.compile(r"\b(?:\(\d{2}\)|\d{2}[ -])\s*9?\s*\d{4}[- ]?\d{4}\b"),
        # CNS: Cartão Nacional de Saúde (15 dígitos).
        # Definitivos iniciam com 1 ou 2; Provisórios com 7, 8 ou 9.
        "CNS": re.compile(r"\b[12789]\d{14}\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    }

    def __init__(self, tag_ini: str = "[", tag_fim: str = "]"):
        """
        Inicializa o motor de anonimização.

        Args:
            tag_ini: Caractere de abertura para tags (padrão: '[').
            tag_fim: Caractere de fechamento para tags (padrão: ']').
        """
        # Marcadores de TAG dinâmicos (selecionados com base no corpus)
        self.TAG_INI = tag_ini
        self.TAG_FIM = tag_fim

        # Delimitador interno dinâmico (baseado nas TAGs da sessão)
        self.DELIMITADOR_INTERNO = tag_ini * 3

        # Mapas de aprendizado (Knowledge Base): {'Texto Original': ID_Unico}
        self.map_nomes_encontrados: dict[str, int] = {}
        self.map_enderecos_encontrados: dict[str, int] = {}
        self.map_outras_info_encontradas: dict[str, int] = {}
        self.map_boilerplates_encontrados: dict[str, int] = {}  # Frases seguras/médicas recorrentes
        self.map_revisoes: dict[str, int] = {}  # Trechos ambíguos marcados para revisão humana futura
        self.map_cpfs_encontrados: dict[str, int] = {}
        self.map_tels_encontrados: dict[str, int] = {}
        self.map_emails_encontrados: dict[str, int] = {}
        self.map_datas_encontradas: dict[str, int] = {}
        self.map_unknown_encontrados: dict[str, int] = {}

        # Índices auxiliares para busca case-insensitive (Lower -> Original)
        self.map_nomes_lower: dict[str, str] = {}
        self.map_enderecos_lower: dict[str, str] = {}
        self.map_outros_lower: dict[str, str] = {}
        self.map_cpfs_lower: dict[str, str] = {}
        self.map_tels_lower: dict[str, str] = {}
        self.map_emails_lower: dict[str, str] = {}
        self.map_datas_lower: dict[str, str] = {}

        # Índice auxiliar para otimização de boilerplates (ignora pontuação)
        self.set_boilerplates_pontuacao_removida: set[str] = set()

        # Contadores para gerar IDs sequenciais únicos para as tags
        self.contador_nomes = 1
        self.contador_enderecos = 1
        self.contador_outros = 1
        self.contador_boilerplates = 1
        self.contador_revisoes = 1
        self.contador_cpfs = 1
        self.contador_tels = 1
        self.contador_emails = 1
        self.contador_datas = 1
        self.contador_unknown = 1

    def converter_tags_para_llm(self, texto: str) -> str:
        """
        Converte TAGs internas para colchetes padrão antes de enviar ao LLM.
        Modelos menores (≤14B) entendem melhor [] que caracteres Unicode especiais.
        """
        if not texto:
            return texto
        return texto.replace(self.TAG_INI, "[").replace(self.TAG_FIM, "]")

    def converter_tags_de_llm(self, texto: str) -> str:
        """
        Converte colchetes da resposta do LLM para TAGs internas.
        Chamado após processar resposta do LLM para manter consistência interna.
        """
        if not texto:
            return texto
        return texto.replace("[", self.TAG_INI).replace("]", self.TAG_FIM)

    def _normalizar_para_comparacao(self, texto: str) -> str:
        """Remove pontuação e converte para minúsculo para comparação flexível."""
        # Mantém apenas letras, números e espaços
        t = re.sub(r"[^\w\s]", "", str(texto).lower())
        return " ".join(t.split())

    def _salvar_nova_tag(self, mapa: dict[str, int], chave: str, contador_attr: str) -> int:
        """
        Adiciona uma chave ao mapa de forma segura, validando contra tags e gerando ID se necessário.
        Retorna o ID (novo ou existente). Retorna -1 se a chave for inválida (contiver tags).
        """
        # Proteção crítica: Nunca permite que uma tag seja chave de um mapa
        if self.TAG_INI in chave or self.TAG_FIM in chave:
            return -1

        if chave not in mapa:
            id_novo = getattr(self, contador_attr)
            mapa[chave] = id_novo
            setattr(self, contador_attr, id_novo + 1)

            # Se for boilerplate, popula o índice otimizado
            if contador_attr == "contador_boilerplates":
                self.set_boilerplates_pontuacao_removida.add(self._normalizar_para_comparacao(chave))

            return id_novo

        return mapa[chave]

    def _registrar_e_get_tag(self, texto: str, tipo_mapa: str) -> str:
        """
        Registra um termo no dicionário correspondente se for novo, e retorna sua tag.
        Garante unicidade global: se o termo já existir em OUTRO mapa, retorna a tag original.
        """
        texto_key = texto.strip()

        # 1. Verificação de Unicidade Global (Evita duplicidade e conflitos na restauração)
        if texto_key in self.map_nomes_encontrados:
            return f"{self.TAG_INI} {self.TAG_NOME} {self.map_nomes_encontrados[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_enderecos_encontrados:
            return f"{self.TAG_INI} {self.TAG_END} {self.map_enderecos_encontrados[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_outras_info_encontradas:
            return f"{self.TAG_INI} {self.TAG_OUTROS} {self.map_outras_info_encontradas[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_cpfs_encontrados:
            return f"{self.TAG_INI} {self.TAG_CPF} {self.map_cpfs_encontrados[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_tels_encontrados:
            return f"{self.TAG_INI} {self.TAG_PHONE} {self.map_tels_encontrados[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_emails_encontrados:
            return f"{self.TAG_INI} {self.TAG_EMAIL} {self.map_emails_encontrados[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_datas_encontradas:
            return f"{self.TAG_INI} {self.TAG_DATE} {self.map_datas_encontradas[texto_key]} {self.TAG_FIM}"
        if texto_key in self.map_unknown_encontrados:
            return f"{self.TAG_INI} {self.TAG_UNKNOWN} {self.map_unknown_encontrados[texto_key]} {self.TAG_FIM}"

        # Lógica de Promoção: Se estava em revisão mas agora tem categoria definitiva, remove da revisão
        if texto_key in self.map_revisoes:
            if tipo_mapa in ["nome", "endereco", "outros", "cpf", "tel", "email", "data", "unknown"]:
                del self.map_revisoes[texto_key]  # Remove do limbo para ser promovido abaixo
            else:
                return f"{self.TAG_INI} {self.TAG_REVISAO} {self.map_revisoes[texto_key]} {self.TAG_FIM}"

        # 2. Seleciona o mapa e contador apropriados para novo registro
        if tipo_mapa == "nome":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_nomes_encontrados,
                "contador_nomes",
                self.TAG_NOME,
                self.map_nomes_lower,
            )
        elif tipo_mapa == "endereco":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_enderecos_encontrados,
                "contador_enderecos",
                self.TAG_END,
                self.map_enderecos_lower,
            )
        elif tipo_mapa == "cpf":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_cpfs_encontrados,
                "contador_cpfs",
                self.TAG_CPF,
                self.map_cpfs_lower,
            )
        elif tipo_mapa == "tel":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_tels_encontrados,
                "contador_tels",
                self.TAG_PHONE,
                self.map_tels_lower,
            )
        elif tipo_mapa == "email":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_emails_encontrados,
                "contador_emails",
                self.TAG_EMAIL,
                self.map_emails_lower,
            )
        elif tipo_mapa == "data":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_datas_encontradas,
                "contador_datas",
                self.TAG_DATE,
                self.map_datas_lower,
            )
        elif tipo_mapa == "revisao":
            mapa, contador_attr, prefixo, mapa_lower = self.map_revisoes, "contador_revisoes", self.TAG_REVISAO, None
        elif tipo_mapa == "unknown":
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_unknown_encontrados,
                "contador_unknown",
                self.TAG_UNKNOWN,
                None,
            )
        else:
            mapa, contador_attr, prefixo, mapa_lower = (
                self.map_outras_info_encontradas,
                "contador_outros",
                self.TAG_OUTROS,
                self.map_outros_lower,
            )

        # Lógica de registro único centralizada
        id_tag = self._salvar_nova_tag(mapa, texto_key, contador_attr)
        if id_tag == -1:
            return texto_key  # Falha de segurança (era uma tag), retorna texto original

        # Atualiza índices auxiliares para detecção de variações
        if mapa_lower is not None:
            mapa_lower[texto_key.lower()] = texto_key

        return f"{self.TAG_INI} {prefixo} {id_tag} {self.TAG_FIM}"

    def _anonimizar_regex_padroes(self, texto: str) -> str:
        """Aplica todas as REGEX_PADROES no texto e substitui por tags específicas."""
        texto_processado = texto

        # Mapeia o nome da chave da REGEX_PADROES para o tipo de mapa esperado no _registrar
        mapa_tipos_regex = {"CPF": "cpf", "DATA": "data", "DATA_EXTENSO": "data", "TEL": "tel", "EMAIL": "email"}

        for nome_regex, regex in Anonimizacao.REGEX_PADROES.items():
            tipo_mapa = mapa_tipos_regex.get(nome_regex, "outros")

            def substituir_match(match, t=tipo_mapa):
                return self._registrar_e_get_tag(match.group(0), t)

            texto_processado = regex.sub(substituir_match, texto_processado)

        return texto_processado

    def _validar_candidato_pii(self, texto: str) -> bool:
        """
        Filtro básico de qualidade para evitar que lixo (ex: '.', 'a') ou
        tags do próprio sistema sejam aprendidas como PII.
        """
        if not isinstance(texto, str):
            return False
        texto_limpo = texto.strip().lower()
        if len(texto_limpo) < 2:
            return False  # Termos muito curtos
        if texto_limpo.isdigit() and len(texto_limpo) < 5:
            return False  # Números curtos isolados

        # Bloqueia tags do sistema para evitar recursividade/corrupção
        if self.TAG_INI in texto or self.TAG_FIM in texto:
            return False

        return True

    def _anonimizar_boilerplates(self, texto: str) -> str:
        """
        Substitui frases inteiras aprendidas como seguras (boilerplates)
        por suas tags correspondentes. Otimiza o prompt para o LLM.
        """
        if not self.map_boilerplates_encontrados:
            return texto

        # Ordena boilerplates por tamanho (maior primeiro) para evitar substituição parcial
        # Ex: Substituir "Paciente com dor" antes de "Paciente"
        frases_ordenadas = sorted(self.map_boilerplates_encontrados.keys(), key=lambda x: (-len(x), x))

        # Cria regex gigante para substituição em massa (Trie approach via Regex)
        # Usa fronteira segura para evitar casar meio de palavra
        chars_palavra = r"a-zA-Z0-9áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ"
        padrao = (
            r"(?<!["
            + chars_palavra
            + r"])("
            + "|".join(re.escape(k) for k in frases_ordenadas)
            + r")(?!["
            + chars_palavra
            + r"])"
        )

        def cb(match):
            txt_match = match.group(0)

            # Usa método seguro que já valida tags
            id_b = self._salvar_nova_tag(self.map_boilerplates_encontrados, txt_match, "contador_boilerplates")

            if id_b == -1:
                return txt_match  # Se for inválido/tag, não substitui
            return f"{self.TAG_INI} {self.TAG_BOILERPLATE} {id_b} {self.TAG_FIM}"

        return re.sub(padrao, cb, texto, flags=re.IGNORECASE)

    def _anonimizar_piis_conhecidos(self, texto: str) -> str:
        """
        Varre o texto procurando por QUALQUER PII que já tenha sido aprendido anteriormente
        em qualquer evolução, garantindo consistência e economia de tokens.
        Preserva a fidelidade do case original registrando variações novas.
        """
        # 1. Mapa exato de tags conhecidas (case sensitive)
        todos_piis = {}
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_NOME} {v} {self.TAG_FIM}") for k, v in self.map_nomes_encontrados.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_END} {v} {self.TAG_FIM}") for k, v in self.map_enderecos_encontrados.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_OUTROS} {v} {self.TAG_FIM}")
            for k, v in self.map_outras_info_encontradas.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_CPF} {v} {self.TAG_FIM}") for k, v in self.map_cpfs_encontrados.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_PHONE} {v} {self.TAG_FIM}") for k, v in self.map_tels_encontrados.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_EMAIL} {v} {self.TAG_FIM}") for k, v in self.map_emails_encontrados.items()
        )
        todos_piis.update(
            (k, f"{self.TAG_INI} {self.TAG_DATE} {v} {self.TAG_FIM}") for k, v in self.map_datas_encontradas.items()
        )

        if not todos_piis:
            return texto

        # Ordenação decrescente de tamanho é crucial para regex 'OR' (|)
        chaves_ordenadas = sorted(todos_piis.keys(), key=lambda x: (-len(x), x))

        # Constrói regex com fronteira segura (Lookaround negativo)
        # Impede match se houver letra ou número antes/depois (incluindo acentos)
        # (?<!...) garante que não tem caractere de palavra antes
        # (?!...) garante que não tem caractere de palavra depois
        chars_palavra = r"a-zA-Z0-9áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ"
        padrao = (
            r"(?<!["
            + chars_palavra
            + r"])("
            + "|".join(re.escape(k) for k in chaves_ordenadas)
            + r")(?!["
            + chars_palavra
            + r"])"
        )

        def cb(match):
            txt_match = match.group(0)
            # Se for o case exato já conhecido, retorna a tag
            if txt_match in todos_piis:
                return todos_piis[txt_match]

            # Se for uma variação (ex: MARIA vs Maria), verifica mapas auxiliares
            txt_lower = txt_match.lower()

            if txt_lower in self.map_nomes_lower:
                return self._registrar_e_get_tag(txt_match, "nome")

            if txt_lower in self.map_enderecos_lower:
                return self._registrar_e_get_tag(txt_match, "endereco")

            if txt_lower in self.map_outros_lower:
                return self._registrar_e_get_tag(txt_match, "outros")

            return txt_match

        return re.sub(padrao, cb, texto, flags=re.IGNORECASE)

    def substituir_pii_lista(
        self,
        texto: str,
        lista_nomes: list[str] | None = None,
        lista_enderecos: list[str] | None = None,
        lista_outros: list[str] | None = None,
    ) -> str:
        """
        Método cirúrgico: Substitui apenas uma lista específica de termos.
        Usado imediatamente após o retorno do LLM para aplicar o que acabou de ser descoberto.
        """
        if not texto:
            return texto

        # Normaliza parâmetros None para listas vazias
        lista_nomes = lista_nomes or []
        lista_enderecos = lista_enderecos or []
        lista_outros = lista_outros or []

        # Mapeia cada termo para sua tag (gerando ID se necessário)
        termos_tags = {}
        for n in lista_nomes:
            termos_tags[n] = self._registrar_e_get_tag(n, "nome")
        for e in lista_enderecos:
            termos_tags[e] = self._registrar_e_get_tag(e, "endereco")
        for o in lista_outros:
            termos_tags[o] = self._registrar_e_get_tag(o, "outros")

        if not termos_tags:
            return texto

        chaves = sorted(termos_tags.keys(), key=lambda x: (-len(x), x))
        chars_palavra = r"a-zA-Z0-9áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ"
        padrao = (
            r"(?<!["
            + chars_palavra
            + r"])("
            + "|".join(re.escape(k) for k in chaves)
            + r")(?!["
            + chars_palavra
            + r"])"
        )

        return re.sub(padrao, lambda m: termos_tags.get(m.group(0), m.group(0)), texto, flags=re.IGNORECASE)

    def preparar_para_llm(self, evolucao: str) -> str:
        """
        Pipeline de limpeza pré-LLM otimizado para contexto máximo.
        Aplica APENAS Boilerplates (redução de frases repetitivas).
        Mantém Regex (Datas, CPF) e PII (Nomes, Endereços) originais.
        """
        # NOTA: Removemos Regex e PII conhecidos para manter o original no LLM (Local)
        return self._anonimizar_boilerplates(evolucao).strip()

    def anonimizar_consolidado(self, evolucao: str) -> str:
        """
        Aplica TODAS as camadas de anonimização para gerar o arquivo final.
        """
        texto = self._anonimizar_regex_padroes(evolucao)
        texto = self._anonimizar_piis_conhecidos(texto)
        return self._anonimizar_boilerplates(texto).strip()

    def atualizar_piis_e_boilerplates(self, texto: str, llm_response: dict, apenas_pii: bool = False):
        """
        Processa a resposta do LLM:
        1. Registra novos PIIs encontrados nos mapas.
        2. Se apenas_pii=False, analisa o que SOBROU do texto e aprende como 'Boilerplate'.
        """
        if llm_response.get("houve_falha", False):
            # Aborta o aprendizado de PIIs e Boilerplates em caso de timeout ou overflow,
            # evitando que PIIs omitidos forçadamente sejam aprendidos como boilerplates.
            return

        try:
            raw = llm_response.get("resposta", "{}")
            dados = json.loads(re.sub(r"```json|```", "", raw).strip())

            # 1. Registro de PIIs
            for n in [t for t in dados.get("map_nomes", []) if self._validar_candidato_pii(t)]:
                self._registrar_e_get_tag(n, "nome")
            for n in [t for t in dados.get("map_enderecos", []) if self._validar_candidato_pii(t)]:
                self._registrar_e_get_tag(n, "endereco")
            for n in [t for t in dados.get("map_outros", []) if self._validar_candidato_pii(t)]:
                self._registrar_e_get_tag(n, "outros")

            # Se a flag apenas_pii estiver ativa, encerra aqui e não aprende boilerplates
            if apenas_pii:
                return

            # 2. Aprendizado de Frases Seguras (Inverted Search com Fragmentação)
            # Tudo que NÃO é PII na frase original é candidato a ser um boilerplate médico.
            todos_piis_llm = sorted(
                list(set(dados.get("map_nomes", []) + dados.get("map_enderecos", []) + dados.get("map_outros", []))),
                key=lambda x: (-len(x), x),
            )

            # Analisa frase por frase
            for frase in re.split(r"(\n|\. )", texto):
                frase_limpa = frase.strip()
                if not frase_limpa:
                    continue

                # Restaura tags existentes para texto puro antes de analisar
                frase_original_completa = self.restaurar_texto_original(frase_limpa)

                # "Explode" a frase usando os PIIs como delimitadores
                # Comparação case-insensitive: "Maria" == "maria" para fins de exclusão de PII
                frase_temp = frase_original_completa
                tem_pii = False
                for pii in todos_piis_llm:
                    if pii.lower() in frase_temp.lower():
                        tem_pii = True
                        frase_temp = re.sub(re.escape(pii), self.DELIMITADOR_INTERNO, frase_temp, flags=re.IGNORECASE)

                # Os fragmentos resultantes são partes "seguras" da frase
                candidatos = frase_temp.split(self.DELIMITADOR_INTERNO) if tem_pii else [frase_temp]

                for fragmento in candidatos:
                    frag = fragmento.strip()
                    # Só aprende se for longo o suficiente (chars E palavras) e não tiver regex (datas, etc) no meio
                    if len(frag) >= self.MIN_CHARS_SAFE_PHRASE and len(frag.split()) >= self.MIN_WORDS_SAFE_PHRASE:
                        if not any(r.search(frag) for r in Anonimizacao.REGEX_PADROES.values()):
                            self._salvar_nova_tag(self.map_boilerplates_encontrados, frag, "contador_boilerplates")
        except json.JSONDecodeError as e:
            logger.log(f"Erro ao decodificar JSON do LLM: {e}", level=LogLevel.ERROR)
        except Exception as e:
            logger.log(f"Erro inesperado ao atualizar PIIs: {e}", level=LogLevel.ERROR)

    # --- N-GRAMS (Detecção Estatística) ---
    def _limpar_id_ngrams(self, id_bruto):
        """Padroniza ID do paciente para agregação."""
        if pd.isna(id_bruto):
            return None
        return re.sub(r"PACIENTE\s*-\s*", "", str(id_bruto).upper()).strip()

    def _limpar_texto_ngrams(self, texto):
        """Simplifica texto para contagem de n-grams (remove pontuação e tags)."""
        if pd.isna(texto):
            return ""

        # Constrói regex dinâmica para remover tags do sistema
        padrao_tags = re.escape(self.TAG_INI) + r".*?" + re.escape(self.TAG_FIM)

        # Substitui tags por delimitador dinâmico para quebrar o fluxo do n-gram
        t = re.sub(padrao_tags, self.DELIMITADOR_INTERNO, str(texto).lower())

        # Remove caracteres especiais mantendo o delimitador
        delim_escaped = re.escape(self.DELIMITADOR_INTERNO[0])
        t = re.sub(r"[^\w\s" + delim_escaped + r"]", " ", t)
        return " ".join(t.split())

    def preparar_corpus_ngrams(
        self, df: pd.DataFrame, coluna_texto=config.COLUNA_TEXTO, coluna_id=config.COLUNA_ID
    ) -> dict[str, list[str]]:
        """
        Pré-processa o dataset agrupando textos por paciente.
        Aplica anonimização prévia (PIIs conhecidos) para evitar contaminação estatística.
        Executado uma única vez antes do loop de N-Grams para otimização.
        """
        logger.log("Pré-processando corpus para análise de N-Grams...", level=LogLevel.NORMAL)
        pacientes_textos = {}
        for pid, txt in zip(df[coluna_id], df[coluna_texto]):
            if pd.isna(txt):
                continue
            pid_limpo = self._limpar_id_ngrams(pid)
            if not pid_limpo:
                continue

            # Aplica conhecimento prévio (Regex, Boilerplates) antes de gerar N-Gram
            # Mantém nomes e endereços para economizar tokens
            txt_anonimizado = self.preparar_para_llm(txt)

            # Limpa o texto usando as regras de tags atuais para o formato de contagem
            txt_limpo = self._limpar_texto_ngrams(txt_anonimizado)

            if pid_limpo not in pacientes_textos:
                pacientes_textos[pid_limpo] = []
            pacientes_textos[pid_limpo].append(txt_limpo)

        return pacientes_textos

    def remover_do_corpus(self, pacientes_textos: dict[str, list[str]], frase_boilerplate: str):
        """
        Remove um boilerplate recém-descoberto do corpus em memória.
        Isso garante que ele não seja contabilizado novamente em iterações de N-Grams menores.
        """
        if not frase_boilerplate:
            return

        # Constrói regex para remover a frase específica, tolerando pontuação
        # Similar a anonimizar_ngram mas troca por marcador neutro &&&
        pals = frase_boilerplate.split()
        padrao = r"\b" + r"[^\w]+".join([re.escape(p) for p in pals]) + r"\b"

        for pid in pacientes_textos:
            novos_textos = []
            for texto in pacientes_textos[pid]:
                # Substitui ocorrências pelo delimitador dinâmico para quebrar a sequência
                novo_texto = re.sub(padrao, self.DELIMITADOR_INTERNO, texto, flags=re.IGNORECASE)
                novos_textos.append(novo_texto)
            pacientes_textos[pid] = novos_textos

    def salvar_novo_boilerplate(self, frase: str):
        """Registra explicitamente uma frase como boilerplate seguro."""
        if not frase:
            return
        self._salvar_nova_tag(self.map_boilerplates_encontrados, frase, "contador_boilerplates")

    def gerar_ngrams_candidatos_boilerplate(
        self,
        pacientes_textos: dict[str, list[str]],
        ngram_size=config.NGRAM_MAX,
        min_pacientes=config.NGRAM_MIN_PACIENTES,
    ):
        """
        Analisa o corpus pré-processado procurando sequências de palavras que se repetem
        em múltiplos pacientes diferentes.
        """
        # Conta N-Grams
        contador_total, contador_pacientes = Counter(), Counter()
        for pid, textos in pacientes_textos.items():
            ngrams_paciente = set()  # Set para contar apenas 1 vez por paciente
            for texto in textos:
                fragmentos = texto.split(self.DELIMITADOR_INTERNO[0])
                for fragmento in fragmentos:
                    palavras = fragmento.split()
                    if len(palavras) >= ngram_size:
                        for i in range(len(palavras) - ngram_size + 1):
                            ngram = " ".join(palavras[i : i + ngram_size])
                            contador_total[ngram] += 1
                            ngrams_paciente.add(ngram)
            for ngram in ngrams_paciente:
                contador_pacientes[ngram] += 1

        # Filtra n-grams candidatos
        ngrams_candidatos = [
            {"n": ngram_size, "frase": ngram, "total": tot, "pacientes": contador_pacientes[ngram]}
            for ngram, tot in contador_total.items()
            if contador_pacientes[ngram] >= min_pacientes
        ]

        # Ordena por: 1º Número de Pacientes Diferentes, 2º Total de Ocorrências
        ngrams_candidatos.sort(key=lambda x: (x["pacientes"], x["total"]), reverse=True)

        if ngrams_candidatos:
            logger.log(
                f"Encontrados {ngram_size}-grams: {len(ngrams_candidatos)} candidatos a boilerplate...", level=LogLevel.NORMAL
            )

        return ngrams_candidatos

    def anonimizar_ngram(self, texto: str, ngram: str) -> str:
        """Substitui uma ocorrência específica de N-Gram por tag de boilerplate."""
        if not texto or not ngram:
            return texto
        pals = ngram.split()
        if not pals:
            return texto
        # Cria regex flexível a pontuação entre as palavras
        padrao = r"\b" + r"[^\w]+".join([re.escape(p) for p in pals]) + r"\b"

        def cb(match):
            txt = match.group(0)
            id_b = self._salvar_nova_tag(self.map_boilerplates_encontrados, txt, "contador_boilerplates")
            if id_b == -1:
                return txt
            return f" {self.TAG_INI} {self.TAG_BOILERPLATE} {id_b} {self.TAG_FIM} "

        return re.sub(padrao, cb, texto, flags=re.IGNORECASE).strip()

    def restaurar_apenas_contexto(self, texto_com_tags: str) -> str:
        """
        Reverte apenas tags de contexto (boilerplate, revisao), mantendo PIIs ocultos.
        Útil para gerar datasets finais onde o texto médico deve estar legível,
        mas os dados pessoais protegidos.
        """
        if not texto_com_tags:
            return ""

        texto = texto_com_tags
        # Busca todas as tags usando as constantes de início e fim
        # Correção: usa \w+ para capturar o ID (que pode vir sujo) e valida depois
        padrao = re.escape(self.TAG_INI) + r"\s*(\w+(?:\s+\w+)*)\s+(\w+)\s*" + re.escape(self.TAG_FIM)
        matches = re.finditer(padrao, texto)
        substituicoes = {}

        # Tipos válidos do sistema para esta função
        tipos_validos = {self.TAG_BOILERPLATE, self.TAG_REVISAO}

        for m in matches:
            tag, tipo, id_str = m.group(0), m.group(1), m.group(2)

            # Validação: ignora tags mal formatadas (tipo inválido ou ID não numérico)
            if tipo not in tipos_validos or not id_str.isdigit():
                continue

            id_tag = int(id_str)

            mapa = self.map_boilerplates_encontrados if tipo == self.TAG_BOILERPLATE else self.map_revisoes
            # Busca inversa: encontra a chave original pelo ID
            original = next((k for k, v in mapa.items() if v == id_tag), None)
            if original:
                substituicoes[tag] = original

        # Aplica reversões
        for tag, orig in substituicoes.items():
            texto = texto.replace(tag, orig)

        return texto

    def marcar_para_revisao(self, texto: str, termo: str) -> str:
        """Marca um trecho com tag de revisão para análise humana futura."""
        if not texto or not termo:
            return texto
        pals = termo.split()
        if not pals:
            return texto
        padrao = r"\b" + r"[^\w]+".join([re.escape(p) for p in pals]) + r"\b"

        def cb(match):
            return self._registrar_e_get_tag(match.group(0), "revisao")

        return re.sub(padrao, cb, texto, flags=re.IGNORECASE).strip()

    def _construir_mapa_invertido(self, mapa: dict[str, int]) -> dict[int, str]:
        """Constrói dicionário invertido {id: texto_original} para lookup O(1)."""
        return {v: k for k, v in mapa.items()}

    def restaurar_texto_original(self, texto_com_tags: str) -> str:
        """
        Reverte TODAS as tags para o texto original.
        Usado para depuração ou para fornecer contexto limpo ao LLM em caso de dúvida.
        """
        if not texto_com_tags:
            return ""
        texto = texto_com_tags
        # Correção: usa \w+ para capturar o ID e valida depois
        padrao = re.escape(self.TAG_INI) + r"\s*(\w+(?:\s+\w+)*)\s+(\w+)\s*" + re.escape(self.TAG_FIM)
        matches = re.finditer(padrao, texto)
        substituicoes = {}

        # Mapeamento de tipos válidos para seus respectivos dicionários
        tipos_mapas = {
            self.TAG_NOME: self.map_nomes_encontrados,
            self.TAG_END: self.map_enderecos_encontrados,
            self.TAG_OUTROS: self.map_outras_info_encontradas,
            self.TAG_BOILERPLATE: self.map_boilerplates_encontrados,
            self.TAG_REVISAO: self.map_revisoes,
            self.TAG_CPF: self.map_cpfs_encontrados,
            self.TAG_PHONE: self.map_tels_encontrados,
            self.TAG_EMAIL: self.map_emails_encontrados,
            self.TAG_DATE: self.map_datas_encontradas,
            self.TAG_UNKNOWN: self.map_unknown_encontrados,
        }

        # Constrói mapas invertidos para lookup O(1)
        mapas_invertidos = {tipo: self._construir_mapa_invertido(mapa) for tipo, mapa in tipos_mapas.items()}

        for m in matches:
            tag, tipo, id_str = m.group(0), m.group(1), m.group(2)

            # Validação: ignora tags mal formatadas (tipo inválido ou ID não numérico)
            if tipo not in mapas_invertidos or not id_str.isdigit():
                continue

            id_tag = int(id_str)
            mapa_inv = mapas_invertidos[tipo]

            if id_tag in mapa_inv:
                substituicoes[tag] = mapa_inv[id_tag]

        for tag, orig in substituicoes.items():
            # Restore suffix-aware: para idades armazenadas como "39 anos", o buffer
            # contém "[ INFO 16 ] anos". Consumir o sufixo literal evita "39 anos anos".
            words = orig.split()
            if len(words) > 1 and words[0].isdigit():
                suffix = ' ' + ' '.join(words[1:])
                if (tag + suffix) in texto:
                    texto = texto.replace(tag + suffix, orig)
                    continue
            texto = texto.replace(tag, orig)
        return texto

    def formatar_tags_para_relatorio(self, texto: str) -> str:
        """
        Simplifica a visualização das tags para o relatório final.
        Converte tags internas dinâmicas (⦃ PERSON 1 ⦄) para colchetes legíveis ([PERSON 1]).
        """
        if not texto:
            return ""
        ini = re.escape(self.TAG_INI)
        fim = re.escape(self.TAG_FIM)

        # Captura conteúdo interno, normaliza espaços e converte para colchetes
        def formatar_tag(match):
            conteudo = match.group(1).strip()
            conteudo = re.sub(r"\s+", " ", conteudo)
            return f"[{conteudo}]"  # Saída sempre com [] para leitura humana/LLM

        padrao = f"{ini}\\s*(.*?)\\s*{fim}"
        texto = re.sub(padrao, formatar_tag, texto)
        return texto

    def verificar_e_aprender_variacao_boilerplate(self, texto: str) -> bool:
        """
        Verifica se o texto é uma variação (ex: pontuação diferente) de um boilerplate já conhecido.
        Se for, aprende a nova forma automaticamente e retorna True.
        Economiza chamadas ao LLM.
        """
        if not texto or len(texto) < self.MIN_CHARS_SAFE_PHRASE:
            return False

        txt_norm = self._normalizar_para_comparacao(texto)
        if txt_norm in self.set_boilerplates_pontuacao_removida:
            # É uma variação segura! Registra a forma original exata para o futuro.
            self.salvar_novo_boilerplate(texto)
            return True

        return False

    def gerar_mascara(self, texto_original: str, chars_mascara: tuple[str, str]) -> str:
        """
        Gera uma versão mascarada do texto mantendo o tamanho exato.
        chars_mascara: (char_pii, char_unknown)
        """
        char_pii, char_unknown = chars_mascara

        # Converte para lista de caracteres para manipulação mutável
        buffer = list(texto_original)
        # Array booleano para marcar posições já ocupadas (evita sobrescrita incorreta)
        ocupado = [False] * len(buffer)

        # 1. Aplica PIIs (Nomes, Endereços, Outros) - Prioridade Alta
        # Agrupa todos os termos PII
        termos_pii = (
            list(self.map_nomes_encontrados.keys())
            + list(self.map_enderecos_encontrados.keys())
            + list(self.map_cpfs_encontrados.keys())
            + list(self.map_tels_encontrados.keys())
            + list(self.map_emails_encontrados.keys())
            + list(self.map_datas_encontradas.keys())
        )

        # Ordena por tamanho decrescente (Greedy) para pegar "São Paulo" antes de "Paulo"
        termos_pii.sort(key=lambda x: (-len(x), x))

        texto_lower = texto_original.lower()

        def aplicar_mascara_termo(termo, caractere):
            t_len = len(termo)
            if t_len == 0:
                return

            start = 0
            t_lower = termo.lower()
            while True:
                idx = texto_lower.find(t_lower, start)
                if idx == -1:
                    break

                # --- VERIFICAÇÃO DE FRONTEIRA DE PALAVRA (Word Boundary) ---
                # Verifica se o caractere anterior é alfanumérico
                is_prefix = False
                if idx > 0:
                    char_ant = texto_original[idx - 1]
                    if char_ant.isalnum() or char_ant in "áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ":
                        is_prefix = True

                # Verifica se o caractere posterior é alfanumérico
                is_suffix = False
                if (idx + t_len) < len(texto_original):
                    char_pos = texto_original[idx + t_len]
                    if char_pos.isalnum() or char_pos in "áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ":
                        is_suffix = True

                # Se for meio de palavra (tem prefixo ou sufixo alfanumérico), ignora
                if is_prefix or is_suffix:
                    start = idx + 1
                    continue

                # Verifica se o intervalo está livre
                if not any(ocupado[idx : idx + t_len]):
                    # Aplica máscara
                    for i in range(idx, idx + t_len):
                        buffer[i] = caractere
                        ocupado[i] = True

                start = idx + 1

        def aplicar_mascara_parcial(termo, n_chars, caractere):
            """Localiza o termo completo (word boundary) mas marca apenas os primeiros n_chars."""
            t_len = len(termo)
            if t_len == 0 or n_chars <= 0:
                return

            start = 0
            t_lower = termo.lower()
            while True:
                idx = texto_lower.find(t_lower, start)
                if idx == -1:
                    break

                is_prefix = False
                if idx > 0:
                    char_ant = texto_original[idx - 1]
                    if char_ant.isalnum() or char_ant in "áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ":
                        is_prefix = True

                is_suffix = False
                if (idx + t_len) < len(texto_original):
                    char_pos = texto_original[idx + t_len]
                    if char_pos.isalnum() or char_pos in "áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ":
                        is_suffix = True

                if is_prefix or is_suffix:
                    start = idx + 1
                    continue

                if not any(ocupado[idx : idx + n_chars]):
                    for i in range(idx, idx + n_chars):
                        buffer[i] = caractere
                        ocupado[i] = True

                start = idx + 1

        for pii in termos_pii:
            aplicar_mascara_termo(pii, char_pii)

        # Processa map_outras_info separadamente: expressões de idade ("39 anos") têm
        # mascaramento parcial — só o número é marcado, " anos" fica visível no buffer.
        # Isso evita falso-positivo em temperaturas e alinha com o gabarito do dataset.
        _age_re = re.compile(r'^(\d+)\s+anos', re.IGNORECASE)
        for termo in sorted(self.map_outras_info_encontradas.keys(), key=lambda x: -len(x)):
            m = _age_re.match(termo)
            if m:
                aplicar_mascara_parcial(termo, len(m.group(1)), char_pii)
            else:
                aplicar_mascara_termo(termo, char_pii)

        # Aplica padrões regex diretamente — independente de mapas pré-populados por workers
        for regex in Anonimizacao.REGEX_PADROES.values():
            for match in regex.finditer(texto_original):
                aplicar_mascara_termo(match.group(0), char_pii)

        # 2. Aplica Unknowns (Dúvidas/Conservador) - Prioridade Baixa
        termos_unk = list(self.map_unknown_encontrados.keys())
        termos_unk.sort(key=lambda x: (-len(x), x))

        for unk in termos_unk:
            aplicar_mascara_termo(unk, char_unknown)

        return "".join(buffer)


# Módulo Anonimizacao carregado com sucesso
