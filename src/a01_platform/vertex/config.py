"""
Módulo de Configuração Específica do Servidor.

Ambiente: vertex
"""

import socket

NOME_SESSAO = "vertex_RC1"
HOSTNAME = "vertex"
HARDWARE_INFO = f"Equipamento: {HOSTNAME}"  # Descrição do ambiente de hardware onde o teste roda.

PASTA_RAIZ = "/caminho/para/mestrado"

SUPABASE_CFG_PATH = f"{PASTA_RAIZ}/config/supabase.cfg"
ARQUIVO_DS_ENTRADA = f"{PASTA_RAIZ}/src/anonymed/datasets/clinical_deid_FULL_SEM_TAGS.json"
DIRETORIO_SAIDA = f"{PASTA_RAIZ}/saida"

ARQUIVO_SAIDA_ORIGINAL = f"{DIRETORIO_SAIDA}/{{timestamp}}_original_{{sessao}}_{{modelo}}.txt"
ARQUIVO_SAIDA_ANONIMIZADO = f"{DIRETORIO_SAIDA}/{{timestamp}}_anonimizado_{{sessao}}_{{modelo}}.txt"

DATASET_INICIO = 0
DATASET_FIM = None  # None processa até o último registro; um inteiro limita a quantidade.

MODELOS_LLM = ['MODELO_PARA_TESTAR']
LLM_TEMPERATURA = 0.1
LLM_NUM_CTX = 24576
LLM_NUM_PREDICT = 20000
OLLAMA_URL = "http://localhost:11434"
LOG_LEVEL = "normal"  # silent | error | warning | normal | verbose

# Modo de recuperação quando o LLM estoura a janela de contexto (done_reason != 'stop').
# True  → Injeta raciocínio truncado no prompt da próxima tentativa (comportamento original).
# False → Falha total imediata: detecção zerada, precisão e recall = 0 (modo experimental).
PROMPT_INJECTION = True

# True  → Envia think=True ao Ollama (modelos com suporte: qwen3, deepseek-r1, etc.).
# False → Envia think=False (modelos sem suporte: gemma3, llama, mistral, etc.).
LLM_THINKING = True

# --- Detecção de Loop (ativo apenas quando PROMPT_INJECTION=True) ---
LOOP_JANELA_CHARS = 100           # Det-1: tamanho da janela de comparação
LOOP_LIMIAR_JANELAS = 3           # Det-1: janelas idênticas consecutivas para detectar
LOOP_REP_N_TAMANHO = 3            # Det-2: tamanho do n-gram (trigram)
LOOP_REP_N_THRESHOLD = 0.5        # Det-2: ratio de repetição acima = loop
LOOP_REP_N_MIN_TOKENS = 150       # Det-2: mínimo de tokens para avaliar
LOOP_REP_N_INTERVALO = 500        # Det-2: reavaliar a cada N chars
LOOP_REP_N_JANELA_TOKENS = 1000  # Det-2: janela de tokens para cálculo rep-n
LOOP_COSINE_JANELA_LINHAS = 10    # Det-3: últimas N linhas a comparar
LOOP_COSINE_THRESHOLD = 0.85      # Det-3: similaridade acima = par similar
LOOP_COSINE_MIN_LINHAS = 20       # Det-3: mínimo de linhas para avaliar
LOOP_COSINE_RATIO_PARES = 0.6     # Det-3: fração de pares similares para detectar loop
LOOP_BLOCO_LIMIAR = 3              # Det-4: blocos idênticos para confirmar loop
LOOP_BLOCO_MIN_LINHAS = 6          # Det-4: linhas significativas mínimas (warmup)
LOOP_BLOCO_MIN_CHARS_LINHA = 10    # Det-4: chars mínimos (strip) para linha significativa

NGRAM_MAX = 101
NGRAM_MIN = 10
NGRAM_MIN_PACIENTES = 5
COLUNA_TEXTO = "descricao"
COLUNA_ID = "prontuario"

VERMELHO = "\033[91m"
VERDE = "\033[92m"
AZUL = "\033[94m"
AMARELO = "\033[93m"
RESET = "\033[0m"

PROMPT_ANONIMIZACAO = '''
Atue como um Auditor Sênior de Privacidade Médica.
Sua tarefa é analisar o texto abaixo e listar EXATAMENTE os trechos que contêm Identificadores Pessoais (PII), retornando um JSON.

### REGRAS CRÍTICAS (LEIA COM ATENÇÃO):
1. **EXTRAÇÃO LITERAL**: O texto extraído deve ser IDÊNTICO ao original, incluindo erros de português, letras maiúsculas/minúsculas e pontuação. NÃO CORRIJA NADA.
2. **TAGS DE CONTEXTO**: O texto pode conter marcadores como [ BOILERPLATE 1 ]. Estes são trechos de texto médico repetitivo que já foram ocultados. Ignore-os.
3. **MEDICINA VS PESSOAS**: Nomes de remédios, doenças, exames e anatomia NÃO são PII. Se parecer um nome (ex: "Dipirona", "AAS"), verifique o contexto clínico.

### O QUE EXTRAIR (CATEGORIAS):
- "map_nomes": Nomes de pessoas reais (Pacientes, Médicos, Familiares). (Ex: "Maria", "Dr. Silva").
- "map_enderecos": Cidades, Ruas, Bairros, Hospitais, Clínicas, Unidades de Saúde.
- "map_outros": CPF, RG, CNS, Telefones, Datas, E-mails, Números de Prontuário, Número de plano de saúde.
- "map_idades": Idade do paciente — extraia a expressão completa ("28 anos", não só "28").
- "map_profissoes": Profissão ou ocupação do paciente (ex: "motorista", "aposentado", "dona de casa"). NÃO inclua especialidades médicas nem profissão da equipe médica.

### O QUE NÃO EXTRAIR (FALSOS POSITIVOS COMUNS):
- NÃO extraia nomes de medicamentos (Amoxicilina, Rivotril).
- NÃO extraia nomes de doenças (Dengue, Alzheimer).
- NÃO extraia siglas médicas (AVC, IAM, PA, FC).
- NÃO extraia unidades de medida ou dosagens (mg, cp, ml).
- NÃO extraia especialidades médicas (Cardiologia, Neurologia, Clínica Médica).
- NÃO extraia a profissão da equipe médica (cardiologista, cirurgião, pediatra).

### EXEMPLO 1 (NOME E LOCAL):
Entrada: "Acompanhante: Leticia Ferreira (tia) residente em Maceió."
Saída Esperada:
{{
    "map_nomes": ["Leticia Ferreira"],
    "map_enderecos": ["Maceió"],
    "map_outros": [],
    "map_idades": [],
    "map_profissoes": [],
    "duvidas_pendentes": []
}}

### EXEMPLO 2 (DADOS MÉDICOS):
Entrada: "O paciente usou AAS infantil e foi ao Hosp. Geral [ BOILERPLATE 5 ]."
Saída Esperada:
{{
    "map_nomes": [],
    "map_enderecos": ["Hosp. Geral"],
    "map_outros": [],
    "map_idades": [],
    "map_profissoes": [],
    "duvidas_pendentes": []
}}
(Nota: "AAS" foi ignorado pois é medicamento).

### EXEMPLO 3 (DÚVIDA):
Entrada: "[ BOILERPLATE 71 ] Rosa para a paciente Maria Silva em 12 de Janeiro."
Saída Esperada:
{{
    "map_nomes": ["Maria Silva"],
    "map_enderecos": [],
    "map_outros": ["12 de Janeiro"],
    "map_idades": [],
    "map_profissoes": [],
    "duvidas_pendentes": ["Rosa"]
}}
(Nota: "Rosa" foi marcado como dúvida pois pode ser o nome de um medicamento ou o nome de uma pessoa dependendo do contexto).

### EXEMPLO 4 (IDADE E PROFISSÃO):
Entrada: "Paciente, 28 anos, motorista, refere dor torácica. Avaliada pela Cardiologia."
Saída Esperada:
{{
    "map_nomes": [],
    "map_enderecos": [],
    "map_outros": [],
    "map_idades": ["28 anos"],
    "map_profissoes": ["motorista"],
    "duvidas_pendentes": []
}}
(Nota: "Cardiologia" é especialidade médica, não PII. A idade é extraída apenas como "28".)

### CAMPO DE DÚVIDA:
Se houver termos que você não consegue decidir se são PII ou Médicos, adicione-os na lista "duvidas_pendentes".

### ENTRADA PARA ANÁLISE:
"""
{texto_entrada}
"""
'''

PROMPT_REVISAO_FINAL = """

Atue como Auditor de Dados. Classifique o termo abaixo, que foi extraído de um contexto médico ambíguo.

Termo: "{texto_entrada}"

Sua decisão define se o termo será ocultado (Privacidade) ou exibido (Contexto Médico).
Adote uma postura CONSERVADORA: Na dúvida, proteja o dado.

Classifique em:
- "nome": Se parecer nome de pessoa.
- "endereco": Se parecer local, cidade ou unidade de saúde.
- "seguro": APENAS se for CERTEZA ABSOLUTA que é: Medicamento, Doença, Sintoma, Anatomia ou Palavra comum da língua portuguesa (ex: "paciente", "dor", "refere").
- "outros": Datas, números, documentos ou QUALQUER INCERTEZA.

Responda APENAS JSON:
{{
    "classificacao": "categoria_escolhida"
}}

"""

PROMPT_LOOP_PENSAMENTO = """

[AVISO OBRIGATÓRIO: Sua tentativa anterior foi interrompida ou atingiu limite de tokens. Abaixo está o seu raciocínio preliminar resgatado. Use-o como base para a continuidade imediata e entregue OBRIGATORIAMENTE apenas o objeto JSON formatado final!]

[RACIOCÍNIO ANTERIOR]
{historico_raciocinio}"""
