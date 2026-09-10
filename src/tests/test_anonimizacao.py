"""
Testes unitários para o módulo Anonimizacao.py

Cobre:
- Regex pré-compiladas (CPF, datas, telefones, CNS, email)
- Sistema de TAGs dinâmicas e conversão LLM
- Registro e unicidade de termos (_registrar_e_get_tag)
- Pipeline de anonimização (preparar_para_llm, anonimizar_consolidado)
- Geração de máscara posicional
- Aprendizado de boilerplates e variações
- Processamento de respostas LLM (atualizar_piis_e_boilerplates)
- Sanitização e validação
"""

from Anonimizacao import Anonimizacao

# =============================================
# Regex Pré-Compiladas
# =============================================


class TestRegexCPF:
    def test_cpf_valido(self):
        assert Anonimizacao.REGEX_PADROES["CPF"].search("CPF: 123.456.789-00")

    def test_cpf_com_ponto_no_verificador(self):
        assert Anonimizacao.REGEX_PADROES["CPF"].search("123.456.789.00")

    def test_texto_sem_cpf(self):
        assert Anonimizacao.REGEX_PADROES["CPF"].search("Paciente refere dor") is None

    def test_numero_parcial_nao_eh_cpf(self):
        assert Anonimizacao.REGEX_PADROES["CPF"].search("123.456") is None


class TestRegexData:
    def test_data_padrao(self):
        assert Anonimizacao.REGEX_PADROES["DATA"].search("Consulta em 15/03/2026")

    def test_data_ano_curto(self):
        assert Anonimizacao.REGEX_PADROES["DATA"].search("Data: 01/01/26")

    def test_data_extenso_completa(self):
        assert Anonimizacao.REGEX_PADROES["DATA_EXTENSO"].search("12 de Janeiro de 2026")

    def test_data_extenso_sem_ano(self):
        assert Anonimizacao.REGEX_PADROES["DATA_EXTENSO"].search("15 de março")

    def test_data_extenso_abreviada(self):
        assert Anonimizacao.REGEX_PADROES["DATA_EXTENSO"].search("3 de fev")


class TestRegexTelefone:
    def test_telefone_sem_parenteses(self):
        """Formato DDD sem parênteses é capturado pela regex."""
        assert Anonimizacao.REGEX_PADROES["TEL"].search("82 99999-1234")

    def test_telefone_com_hifen(self):
        """Formato DDD-número é capturado."""
        assert Anonimizacao.REGEX_PADROES["TEL"].search("82-99999-1234")

    def test_telefone_com_parenteses_nao_captura(self):
        """Formato (DDD) não é capturado pela regex atual (\\b antes de parênteses)."""
        # Documentação: a regex atual usa \\b que não faz match antes de '('
        # Isso é comportamento CONHECIDO, não um bug
        assert Anonimizacao.REGEX_PADROES["TEL"].search("(82) 99999-1234") is None


class TestRegexCNS:
    def test_cns_definitivo(self):
        assert Anonimizacao.REGEX_PADROES["CNS"].search("CNS: 198765432109876")

    def test_cns_provisorio(self):
        assert Anonimizacao.REGEX_PADROES["CNS"].search("798765432109876")


class TestRegexEmail:
    def test_email_valido(self):
        assert Anonimizacao.REGEX_PADROES["EMAIL"].search("contato@hospital.com.br")


# =============================================
# TAGs Dinâmicas e Conversão LLM
# =============================================


class TestTagsDinamicas:
    def test_tags_customizadas_no_init(self):
        a = Anonimizacao(tag_ini="⦃", tag_fim="⦄")
        assert a.TAG_INI == "⦃"
        assert a.TAG_FIM == "⦄"

    def test_delimitador_interno_baseado_em_tag(self):
        a = Anonimizacao(tag_ini="⦃", tag_fim="⦄")
        assert a.DELIMITADOR_INTERNO == "⦃⦃⦃"

    def test_converter_tags_para_llm(self, anonimizacao):
        texto = "⦃ PERSON 1 ⦄ refere dor"
        resultado = anonimizacao.converter_tags_para_llm(texto)
        assert resultado == "[ PERSON 1 ] refere dor"

    def test_converter_tags_de_llm(self, anonimizacao):
        texto = "[ PERSON 1 ] refere dor"
        resultado = anonimizacao.converter_tags_de_llm(texto)
        assert resultado == "⦃ PERSON 1 ⦄ refere dor"

    def test_converter_texto_vazio(self, anonimizacao):
        assert anonimizacao.converter_tags_para_llm("") == ""
        assert anonimizacao.converter_tags_de_llm("") == ""

    def test_converter_none(self, anonimizacao):
        assert anonimizacao.converter_tags_para_llm(None) is None
        assert anonimizacao.converter_tags_de_llm(None) is None


# =============================================
# Registro de Termos (_registrar_e_get_tag)
# =============================================


class TestRegistroTermos:
    def test_registrar_nome(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        assert "PERSON" in tag
        assert "1" in tag
        assert anonimizacao.TAG_INI in tag

    def test_registrar_endereco(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("Maceió", "endereco")
        assert "LOCATION" in tag

    def test_registrar_outros(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("RG 12345", "outros")
        assert "INFO" in tag

    def test_registrar_revisao(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("Rosa", "revisao")
        assert "REVIEW" in tag

    def test_registrar_unknown(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("termo ambíguo", "unknown")
        assert "UNKNOWN" in tag

    def test_mesmo_termo_retorna_mesmo_id(self, anonimizacao):
        tag1 = anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        tag2 = anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        assert tag1 == tag2

    def test_termos_diferentes_ids_diferentes(self, anonimizacao):
        tag1 = anonimizacao._registrar_e_get_tag("Maria", "nome")
        tag2 = anonimizacao._registrar_e_get_tag("João", "nome")
        assert tag1 != tag2

    def test_unicidade_global_entre_mapas(self, anonimizacao):
        """Se 'Maria' já é nome, registrar como endereço deve retornar a tag de nome."""
        tag_nome = anonimizacao._registrar_e_get_tag("Maria", "nome")
        tag_end = anonimizacao._registrar_e_get_tag("Maria", "endereco")
        assert tag_nome == tag_end
        assert "PERSON" in tag_end  # Mantém o tipo original

    def test_promocao_de_revisao_para_nome(self, anonimizacao):
        """Termo em revisão deve ser promovido para nome quando classificado."""
        anonimizacao._registrar_e_get_tag("Rosa", "revisao")
        assert "Rosa" in anonimizacao.map_revisoes
        tag = anonimizacao._registrar_e_get_tag("Rosa", "nome")
        assert "PERSON" in tag
        assert "Rosa" not in anonimizacao.map_revisoes  # Removido da revisão

    def test_tag_com_tag_ini_retorna_texto_original(self, anonimizacao):
        """Deve rejeitar chaves que contenham TAG_INI/TAG_FIM (proteção anti-recursão)."""
        resultado = anonimizacao._salvar_nova_tag(anonimizacao.map_nomes_encontrados, "⦃ PERSON 1 ⦄", "contador_nomes")
        assert resultado == -1

    def test_strip_de_espacos(self, anonimizacao):
        """Deve fazer strip dos espaços ao registrar."""
        tag1 = anonimizacao._registrar_e_get_tag("Maria", "nome")
        tag2 = anonimizacao._registrar_e_get_tag("  Maria  ", "nome")
        assert tag1 == tag2

    def test_contadores_incrementam(self, anonimizacao):
        """Contadores devem incrementar a cada novo registro."""
        assert anonimizacao.contador_nomes == 1
        anonimizacao._registrar_e_get_tag("Ana", "nome")
        assert anonimizacao.contador_nomes == 2
        anonimizacao._registrar_e_get_tag("Bia", "nome")
        assert anonimizacao.contador_nomes == 3


# =============================================
# Pipeline de Anonimização
# =============================================


class TestPipelineAnonimizacao:
    def test_preparar_para_llm_aplica_boilerplates(self, anonimizacao_com_dados):
        texto = "Paciente em bom estado geral, com dor leve."
        resultado = anonimizacao_com_dados.preparar_para_llm(texto)
        assert "BOILERPLATE" in resultado
        assert "dor leve" in resultado

    def test_preparar_para_llm_nao_aplica_nomes(self, anonimizacao_com_dados):
        """preparar_para_llm NÃO substitui nomes (mantém para o LLM ver)."""
        texto = "Maria Silva refere dor."
        resultado = anonimizacao_com_dados.preparar_para_llm(texto)
        assert "Maria Silva" in resultado

    def test_anonimizar_consolidado_aplica_tudo(self, anonimizacao_com_dados):
        texto = "Maria Silva de Maceió, CPF 123.456.789-00. Tel (82) 99999-0000."
        resultado = anonimizacao_com_dados.anonimizar_consolidado(texto)
        assert "Maria Silva" not in resultado
        assert "Maceió" not in resultado
        assert "123.456.789-00" not in resultado
        assert "PERSON" in resultado
        assert "LOCATION" in resultado

    def test_anonimizar_consolidado_preserva_texto_medico(self, anonimizacao_com_dados):
        """Termos médicos NÃO devem ser substituídos."""
        texto = "Prescrito Dipirona 500mg. PA 120/80mmHg."
        resultado = anonimizacao_com_dados.anonimizar_consolidado(texto)
        assert "Dipirona" in resultado
        assert "120/80" in resultado

    def test_regex_captura_data_no_consolidado(self, anonimizacao_com_dados):
        texto = "Consulta em 15/03/2026."
        resultado = anonimizacao_com_dados.anonimizar_consolidado(texto)
        assert "15/03/2026" not in resultado
        assert "DATE" in resultado


# =============================================
# Geração de Máscara Posicional
# =============================================


class TestGerarMascara:
    def test_mascara_mesmo_tamanho_do_original(self, anonimizacao_com_dados):
        texto = "Maria Silva mora em Maceió."
        mascara = anonimizacao_com_dados.gerar_mascara(texto, ("█", "?"))
        assert len(mascara) == len(texto)

    def test_mascara_substitui_pii_por_char(self, anonimizacao_com_dados):
        texto = "Maria Silva é paciente."
        mascara = anonimizacao_com_dados.gerar_mascara(texto, ("█", "?"))
        # As posições de "Maria Silva" devem ser '█'
        assert mascara[0:12].count("█") == len("Maria Silva")

    def test_mascara_preserva_texto_normal(self, anonimizacao_com_dados):
        texto = "Paciente refere dor."
        mascara = anonimizacao_com_dados.gerar_mascara(texto, ("█", "?"))
        # Sem PII, a máscara deve ser idêntica ao original
        assert mascara == texto

    def test_mascara_unknown_usa_char_unk(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("termo ambíguo", "unknown")
        texto = "Texto com termo ambíguo aqui."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        # "termo ambíguo" deve ser mascarado com '?'
        inicio = texto.find("termo ambíguo")
        for i in range(inicio, inicio + len("termo ambíguo")):
            assert mascara[i] == "?"

    def test_mascara_word_boundary(self, anonimizacao):
        """Não deve mascarar 'Ana' dentro de 'Anamnese'."""
        anonimizacao._registrar_e_get_tag("Ana", "nome")
        texto = "Anamnese normal. Paciente Ana está bem."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        # "Anamnese" NÃO deve ser mascarado
        assert mascara[:8] == "Anamnese"
        # "Ana" no final deve ser mascarado
        idx_ana = texto.find("Ana está")
        assert mascara[idx_ana : idx_ana + 3] == "███"


# =============================================
# Boilerplates
# =============================================


class TestBoilerplates:
    def test_salvar_novo_boilerplate(self, anonimizacao):
        anonimizacao.salvar_novo_boilerplate("Paciente em bom estado geral")
        assert "Paciente em bom estado geral" in anonimizacao.map_boilerplates_encontrados

    def test_verificar_variacao_boilerplate(self, anonimizacao):
        """Variação sem pontuação deve ser reconhecida."""
        anonimizacao.salvar_novo_boilerplate("Paciente em bom estado geral.")
        resultado = anonimizacao.verificar_e_aprender_variacao_boilerplate("Paciente em bom estado geral")
        assert resultado is True

    def test_variacao_curta_nao_aprende(self, anonimizacao):
        """Frases muito curtas não devem ser aprendidas como variação."""
        anonimizacao.salvar_novo_boilerplate("Paciente em bom estado geral")
        resultado = anonimizacao.verificar_e_aprender_variacao_boilerplate("ok")
        assert resultado is False


# =============================================
# Processamento de Resposta LLM
# =============================================


class TestAtualizarPiis:
    def test_aprende_nomes_da_resposta(self, anonimizacao):
        resposta = {"resposta": '{"map_nomes": ["Dr. Silva"], "map_enderecos": [], "map_outros": []}'}
        anonimizacao.atualizar_piis_e_boilerplates("Dr. Silva atendeu.", resposta)
        assert "Dr. Silva" in anonimizacao.map_nomes_encontrados

    def test_aprende_enderecos_da_resposta(self, anonimizacao):
        resposta = {"resposta": '{"map_nomes": [], "map_enderecos": ["Hospital Santa Casa"], "map_outros": []}'}
        anonimizacao.atualizar_piis_e_boilerplates("Internado no Hospital Santa Casa.", resposta)
        assert "Hospital Santa Casa" in anonimizacao.map_enderecos_encontrados

    def test_valida_candidato_pii_curto(self, anonimizacao):
        """Termos com menos de 2 caracteres devem ser rejeitados."""
        assert anonimizacao._validar_candidato_pii(".") is False
        assert anonimizacao._validar_candidato_pii("a") is False

    def test_valida_candidato_pii_numero_curto(self, anonimizacao):
        """Números curtos isolados devem ser rejeitados."""
        assert anonimizacao._validar_candidato_pii("123") is False

    def test_valida_candidato_pii_com_tag(self, anonimizacao):
        """Termos contendo TAG_INI/TAG_FIM devem ser rejeitados."""
        assert anonimizacao._validar_candidato_pii("⦃ PERSON 1 ⦄") is False

    def test_valida_candidato_pii_valido(self, anonimizacao):
        assert anonimizacao._validar_candidato_pii("Maria Silva") is True

    def test_json_invalido_nao_quebra(self, anonimizacao):
        """JSON inválido do LLM não deve lançar exceção."""
        resposta = {"resposta": "isso não é JSON"}
        anonimizacao.atualizar_piis_e_boilerplates("Texto qualquer.", resposta)
        # Apenas não deve lançar exceção


# =============================================
# Restaurar Texto Original
# =============================================


class TestRestaurarTexto:
    def test_restaurar_nome(self, anonimizacao):
        tag = anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        texto = f"A {tag} está bem."
        restaurado = anonimizacao.restaurar_texto_original(texto)
        assert "Maria Silva" in restaurado

    def test_restaurar_boilerplate(self, anonimizacao):
        anonimizacao.salvar_novo_boilerplate("Paciente em bom estado geral")
        tag_id = anonimizacao.map_boilerplates_encontrados["Paciente em bom estado geral"]
        texto = f"⦃ BOILERPLATE {tag_id} ⦄, sem queixas."
        restaurado = anonimizacao.restaurar_apenas_contexto(texto)
        assert "Paciente em bom estado geral" in restaurado

    def test_restaurar_texto_vazio(self, anonimizacao):
        assert anonimizacao.restaurar_texto_original("") == ""
        assert anonimizacao.restaurar_apenas_contexto("") == ""


# =============================================
# Formatar Tags para Relatório
# =============================================


class TestFormatarTags:
    def test_formatar_tag_para_colchetes(self, anonimizacao):
        texto = "⦃ PERSON 1 ⦄ refere dor"
        resultado = anonimizacao.formatar_tags_para_relatorio(texto)
        assert resultado == "[PERSON 1] refere dor"

    def test_formatar_texto_sem_tags(self, anonimizacao):
        texto = "Texto simples sem tags"
        resultado = anonimizacao.formatar_tags_para_relatorio(texto)
        assert resultado == texto

    def test_formatar_texto_vazio(self, anonimizacao):
        assert anonimizacao.formatar_tags_para_relatorio("") == ""


# =============================================
# _anonimizar_piis_conhecidos
# =============================================


class TestAnonimizarPiisConhecidos:
    def test_substitui_nome_conhecido(self, anonimizacao_com_dados):
        texto = "A Maria Silva refere dor."
        resultado = anonimizacao_com_dados._anonimizar_piis_conhecidos(texto)
        assert "Maria Silva" not in resultado
        assert "PERSON" in resultado

    def test_preserva_texto_sem_pii(self, anonimizacao_com_dados):
        texto = "Prescrito Dipirona 500mg."
        resultado = anonimizacao_com_dados._anonimizar_piis_conhecidos(texto)
        assert resultado == texto

    def test_nao_substitui_substring(self, anonimizacao):
        """Não deve substituir 'Ana' dentro de 'Anamnese'."""
        anonimizacao._registrar_e_get_tag("Ana", "nome")
        texto = "Anamnese: Ana refere dor."
        resultado = anonimizacao._anonimizar_piis_conhecidos(texto)
        assert "Anamnese" in resultado  # Preservado
        # "Ana" isolada deve ser substituída
        assert resultado.count("PERSON") == 1

    def test_mapa_vazio_retorna_texto_original(self, anonimizacao):
        texto = "Texto sem PIIs conhecidos."
        resultado = anonimizacao._anonimizar_piis_conhecidos(texto)
        assert resultado == texto


# =============================================
# substituir_pii_lista
# =============================================


class TestSubstituirPiiLista:
    def test_substitui_nomes_da_lista(self, anonimizacao):
        texto = "Dr. João atendeu Maria."
        resultado = anonimizacao.substituir_pii_lista(texto, lista_nomes=["Dr. João", "Maria"])
        assert "Dr. João" not in resultado
        assert "Maria" not in resultado
        assert "PERSON" in resultado

    def test_substitui_endereco(self, anonimizacao):
        texto = "Mora em Maceió."
        resultado = anonimizacao.substituir_pii_lista(texto, lista_enderecos=["Maceió"])
        assert "Maceió" not in resultado
        assert "LOCATION" in resultado

    def test_listas_none_nao_quebra(self, anonimizacao):
        texto = "Texto qualquer."
        resultado = anonimizacao.substituir_pii_lista(texto)
        assert resultado == texto

    def test_texto_vazio_retorna_vazio(self, anonimizacao):
        assert anonimizacao.substituir_pii_lista("") == ""


# =============================================
# anonimizar_ngram
# =============================================


class TestAnonimizarNgram:
    def test_ngram_simples(self, anonimizacao):
        texto = "Paciente em bom estado geral apresenta melhora."
        resultado = anonimizacao.anonimizar_ngram(texto, "Paciente em bom estado geral")
        assert "BOILERPLATE" in resultado
        assert "melhora" in resultado

    def test_ngram_inexistente(self, anonimizacao):
        texto = "Texto sem o ngram alvo."
        resultado = anonimizacao.anonimizar_ngram(texto, "frase inexistente aqui")
        assert resultado == texto

    def test_texto_vazio(self, anonimizacao):
        assert anonimizacao.anonimizar_ngram("", "ngram") == ""

    def test_ngram_vazio(self, anonimizacao):
        assert anonimizacao.anonimizar_ngram("texto", "") == "texto"


# =============================================
# marcar_para_revisao
# =============================================


class TestMarcarParaRevisao:
    def test_marca_termo(self, anonimizacao):
        texto = "A Rosa refere dor."
        resultado = anonimizacao.marcar_para_revisao(texto, "Rosa")
        assert "REVIEW" in resultado
        assert "Rosa" in anonimizacao.map_revisoes

    def test_preserva_contexto(self, anonimizacao):
        texto = "Paciente Rosa está bem."
        resultado = anonimizacao.marcar_para_revisao(texto, "Rosa")
        assert "Paciente" in resultado
        assert "está bem" in resultado

    def test_texto_vazio(self, anonimizacao):
        assert anonimizacao.marcar_para_revisao("", "termo") == ""

    def test_termo_vazio(self, anonimizacao):
        assert anonimizacao.marcar_para_revisao("texto", "") == "texto"


# =============================================
# Máscara: PIIs de Regex (CPF, Data, Telefone, Email)
# =============================================


class TestMascaraRegex:
    def test_mascara_cpf(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("123.456.789-00", "cpf")
        texto = "CPF do paciente 123.456.789-00 registrado."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        assert len(mascara) == len(texto)
        inicio = texto.find("123.456.789-00")
        for i in range(inicio, inicio + len("123.456.789-00")):
            assert mascara[i] == "█"

    def test_mascara_telefone(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("(11) 98765-4321", "tel")
        texto = "Contato: (11) 98765-4321 para emergências."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("(11) 98765-4321")
        for i in range(inicio, inicio + len("(11) 98765-4321")):
            assert mascara[i] == "█"

    def test_mascara_email(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("paciente@email.com", "email")
        texto = "Email: paciente@email.com cadastrado."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("paciente@email.com")
        for i in range(inicio, inicio + len("paciente@email.com")):
            assert mascara[i] == "█"

    def test_mascara_data(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("15/03/2026", "data")
        texto = "Data atendimento: 15/03/2026 na clínica."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("15/03/2026")
        for i in range(inicio, inicio + len("15/03/2026")):
            assert mascara[i] == "█"


# =============================================
# Máscara: Múltiplas Ocorrências e Case
# =============================================


class TestMascaraMultiplasOcorrencias:
    def test_mesmo_pii_aparece_duas_vezes(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("Maria", "nome")
        texto = "Maria refere dor. Acompanhante: Maria (mãe)."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        assert mascara.count("█") == len("Maria") * 2

    def test_case_insensitive_mascara(self, anonimizacao):
        """PII registrado como 'Maria' deve mascarar 'maria' também."""
        anonimizacao._registrar_e_get_tag("Maria", "nome")
        texto = "A maria está bem."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("maria")
        for i in range(inicio, inicio + len("maria")):
            assert mascara[i] == "█"


# =============================================
# Máscara: Sobreposição de PIIs
# =============================================


class TestMascaraSobreposicao:
    def test_pii_longo_tem_prioridade(self, anonimizacao):
        """'Maria Silva' (mais longo) deve ser mascarado antes de 'Maria'."""
        anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        anonimizacao._registrar_e_get_tag("Maria", "nome")
        texto = "Paciente Maria Silva recebeu alta."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("Maria Silva")
        # Toda a região de "Maria Silva" deve ser mascarada
        for i in range(inicio, inicio + len("Maria Silva")):
            assert mascara[i] == "█"

    def test_pii_e_unknown_juntos(self, anonimizacao):
        """PII tem prioridade sobre Unknown na mesma posição."""
        anonimizacao._registrar_e_get_tag("Maria", "nome")
        anonimizacao._registrar_e_get_tag("Maria", "unknown")  # Será ignorado (já em nomes)
        texto = "A Maria está bem."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("Maria")
        for i in range(inicio, inicio + len("Maria")):
            assert mascara[i] == "█"  # PII, não unknown


# =============================================
# BUG FIX: _anonimizar_piis_conhecidos inclui regex PIIs
# =============================================


class TestAnonimizarPiisConhecidosRegex:
    def test_cpf_conhecido_substituido(self, anonimizacao):
        """CPF registrado em registro anterior deve ser substituído por tag."""
        anonimizacao._registrar_e_get_tag("123.456.789-00", "cpf")
        texto = "O CPF 123.456.789-00 foi informado."
        resultado = anonimizacao._anonimizar_piis_conhecidos(texto)
        assert "123.456.789-00" not in resultado
        assert "CPF" in resultado

    def test_email_conhecido_substituido(self, anonimizacao):
        anonimizacao._registrar_e_get_tag("teste@email.com", "email")
        texto = "Contato: teste@email.com para retorno."
        resultado = anonimizacao._anonimizar_piis_conhecidos(texto)
        assert "teste@email.com" not in resultado
        assert "EMAIL" in resultado


# =============================================
# BUG-2: gerar_mascara independente dos mapas regex (fix: regex direta)
# =============================================


class TestMascaraRegexDireta:
    """
    Verifica que gerar_mascara aplica REGEX_PADROES diretamente sobre o texto,
    sem depender de map_cpfs/map_tels/map_emails/map_datas pré-populados.

    Contexto: pandarallel usa fork — side effects de anonimizar_consolidado nos
    workers não voltam ao processo principal. gerar_mascara não pode depender
    de mapas populados por workers.
    """

    def test_cpf_mascarado_sem_mapa_populado(self, anonimizacao):
        """CPF no texto raw deve ser mascarado mesmo com map_cpfs_encontrados vazio."""
        assert not anonimizacao.map_cpfs_encontrados, "pré-condição: mapa deve estar vazio"
        texto = "CPF do paciente: 123.456.789-00. Alta hospitalar."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("123.456.789-00")
        for i in range(inicio, inicio + len("123.456.789-00")):
            assert mascara[i] == "█", f"posição {i}: CPF deve ser mascarado pela regex direta"

    def test_email_mascarado_sem_mapa_populado(self, anonimizacao):
        """E-mail no texto raw deve ser mascarado mesmo com map_emails_encontrados vazio."""
        assert not anonimizacao.map_emails_encontrados, "pré-condição: mapa deve estar vazio"
        texto = "Contato: paciente@hospital.com.br para seguimento."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("paciente@hospital.com.br")
        for i in range(inicio, inicio + len("paciente@hospital.com.br")):
            assert mascara[i] == "█", f"posição {i}: e-mail deve ser mascarado pela regex direta"

    def test_telefone_mascarado_sem_mapa_populado(self, anonimizacao):
        """Telefone (formato sem parênteses) deve ser mascarado mesmo com map_tels vazio."""
        assert not anonimizacao.map_tels_encontrados, "pré-condição: mapa deve estar vazio"
        texto = "Telefone para contato: 82 99999-1234 urgente."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("82 99999-1234")
        for i in range(inicio, inicio + len("82 99999-1234")):
            assert mascara[i] == "█", f"posição {i}: telefone deve ser mascarado pela regex direta"

    def test_regex_e_mapa_nomeado_coexistem(self, anonimizacao):
        """PII nomeado (via mapa LLM) e PII regex (scan direto) coexistem sem conflito."""
        anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        texto = "Paciente Maria Silva. CPF: 123.456.789-00. Alta."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio_nome = texto.find("Maria Silva")
        for i in range(inicio_nome, inicio_nome + len("Maria Silva")):
            assert mascara[i] == "█", "nome via mapa deve ser mascarado"
        inicio_cpf = texto.find("123.456.789-00")
        for i in range(inicio_cpf, inicio_cpf + len("123.456.789-00")):
            assert mascara[i] == "█", "CPF via regex direta deve ser mascarado"

    def test_comprimento_preservado_com_regex_direta(self, anonimizacao):
        """A máscara deve ter exatamente o mesmo comprimento do texto original."""
        texto = "Paciente com CPF 123.456.789-00 e email teste@clinica.com."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        assert len(mascara) == len(texto)


# =============================================
# BUG-9: PII com casing diferente não deve virar boilerplate
# =============================================


class TestBug9CaseInsensitivePiiExclusion:
    """
    BUG-9: Se o LLM retorna "Letícia Ferreira" mas o texto bruto tem
    "letícia ferreira", a exclusão case-sensitive falhava e o fragmento
    contendo o nome podia ser aprendido como boilerplate.

    Princípio: Comparações → case-insensitive; Restaurações → case-sensitive.
    """

    def test_pii_case_mismatch_nao_vira_boilerplate(self, anonimizacao):
        """PII com casing diferente do texto original NÃO deve ser aprendido como boilerplate."""
        # LLM retorna nomes em Title Case
        resposta = {
            "resposta": '{"map_nomes": ["Letícia Ferreira"], "map_enderecos": ["Maceió"], "map_outros": []}'
        }
        # Texto bruto tem casing diferente (caixa baixa, como digitado no prontuário)
        texto = "acompanhante letícia ferreira residente em maceió e outros dados do prontuário"
        anonimizacao.atualizar_piis_e_boilerplates(texto, resposta)

        # O nome NÃO deve ter sido aprendido como boilerplate
        for boilerplate in anonimizacao.map_boilerplates_encontrados:
            assert "letícia" not in boilerplate.lower(), (
                f"Nome 'letícia ferreira' vazou para boilerplate: '{boilerplate}'"
            )
            assert "maceió" not in boilerplate.lower(), (
                f"Endereço 'maceió' vazou para boilerplate: '{boilerplate}'"
            )

    def test_restauracao_preserva_casing_original(self, anonimizacao):
        """Restauração deve retornar o texto exatamente como registrado (case-sensitive)."""
        tag = anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        texto_com_tag = f"A {tag} estava bem."
        restaurado = anonimizacao.restaurar_texto_original(texto_com_tag)
        # Deve restaurar com o casing exato do registro ("Maria Silva", não "maria silva")
        assert "Maria Silva" in restaurado
        assert "maria silva" not in restaurado.replace("Maria Silva", "")

    def test_mascara_case_insensitive(self, anonimizacao):
        """PII registrado como 'José' deve mascarar 'JOSÉ' no texto."""
        anonimizacao._registrar_e_get_tag("José", "nome")
        texto = "Paciente JOSÉ refere dor."
        mascara = anonimizacao.gerar_mascara(texto, ("█", "?"))
        inicio = texto.find("JOSÉ")
        for i in range(inicio, inicio + len("JOSÉ")):
            assert mascara[i] == "█", f"posição {i}: 'JOSÉ' deve ser mascarado (PII registrado como 'José')"

