create table public.p02_brkga_logs_llm (
  id bigint generated always as identity not null,
  resultado_id bigint not null,
  repeticao smallint not null,
  texto_pre_llm text null,
  raciocinio text null,
  resposta_bruta text null,
  llm_raw_response text null,
  status_erro character varying(30) null,
  houve_falha boolean not null default false,
  created_at timestamp without time zone not null default now(),
  done_reason character varying(255) null,
  total_duration_sec double precision null,
  load_duration_sec double precision null,
  prompt_eval_count integer null,
  prompt_eval_duration_sec double precision null,
  eval_count integer null,
  eval_duration_sec double precision null,
  eval_tokens_per_second double precision null,
  constraint p02_brkga_logs_llm_pkey primary key (id),
  constraint p02_brkga_logs_llm_resultado_fkey foreign KEY (resultado_id) references p02_brkga_resultados (id) on update CASCADE on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_brkga_logs_llm_resultado on public.p02_brkga_logs_llm using btree (resultado_id) TABLESPACE pg_default;