-- MIGRATION: Indicadores Ollama em p02_logs_execucao + correção de tipo em p02_sessoes
--

-- 1. Novos indicadores em p02_logs_execucao
ALTER TABLE p02_logs_execucao
  ADD COLUMN IF NOT EXISTS prompt_eval_count          INTEGER,        -- tokens reais do prompt (API Ollama)
  ADD COLUMN IF NOT EXISTS eval_count                 INTEGER,        -- tokens reais gerados (API Ollama)
  ADD COLUMN IF NOT EXISTS eval_tokens_per_second     REAL,           -- throughput de geração (tokens/s)
  ADD COLUMN IF NOT EXISTS total_duration_ollama_sec  REAL,           -- tempo total medido pelo Ollama
  ADD COLUMN IF NOT EXISTS load_duration_sec          REAL,           -- tempo de carregamento do modelo
  ADD COLUMN IF NOT EXISTS prompt_eval_duration_sec   REAL,           -- tempo de prefill (avaliação do prompt)
  ADD COLUMN IF NOT EXISTS eval_duration_sec          REAL,           -- tempo de decoding (geração da resposta)
  ADD COLUMN IF NOT EXISTS done_reason                VARCHAR(30),    -- "stop" | "length" | outro
  ADD COLUMN IF NOT EXISTS qtde_tentativas            SMALLINT DEFAULT 1,  -- 1 ou 2 na fase linha_a_linha
  ADD COLUMN IF NOT EXISTS chars_reducao_boilerplate  INTEGER;        -- chars economizados por boilerplates

-- 2. Raciocínio e resposta bruta para monitoramento de loops de raciocínio
ALTER TABLE p02_logs_execucao
  ADD COLUMN IF NOT EXISTS raciocinio    TEXT,           -- CoT thinking do modelo (thinking tokens)
  ADD COLUMN IF NOT EXISTS resposta_bruta TEXT;          -- texto bruto retornado pelo LLM antes do parse JSON

-- 3b. status_erro por tentativa (espelha p02_brkga_logs_llm; log per-attempt)
ALTER TABLE p02_logs_execucao
  ADD COLUMN IF NOT EXISTS status_erro VARCHAR(30);      -- "erro_stop_reason" | "timeout" | "erro" | null

-- 3. Correção de tipo: ultima_atualizacao TEXT → TIMESTAMP WITH TIME ZONE
--    O formato existente ("2026-05-06 19:55:37") é compatível com o CAST.
ALTER TABLE p02_sessoes
  ALTER COLUMN ultima_atualizacao TYPE TIMESTAMP WITH TIME ZONE
  USING ultima_atualizacao::TIMESTAMP WITH TIME ZONE;
