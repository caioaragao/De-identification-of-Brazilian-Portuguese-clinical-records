create table public.p02_brkga_comparar_deteccoes_loop (
  id bigint generated always as identity not null,
  log_llm_id bigint not null,
  sessao_brkga_id integer not null,
  tamanho_raciocinio integer null,

  -- Det-1/2/3 (abordagem do artigo, em cascata)
  artigo_detectou boolean not null default false,
  artigo_tipo character varying(50) null,
  artigo_posicao integer null,
  artigo_ratio_aproveitado double precision null,

  -- Det-4 (blocos de linhas, isolado)
  det4_detectou boolean not null default false,
  det4_tipo character varying(50) null,
  det4_posicao integer null,
  det4_ratio_aproveitado double precision null,

  -- Comparação derivada
  ambos_detectaram boolean not null default false,
  apenas_artigo boolean not null default false,
  apenas_det4 boolean not null default false,
  nenhum_detectou boolean not null default false,
  diferenca_posicao integer null,
  detector_mais_preciso character varying(10) null,

  created_at timestamp without time zone not null default now(),

  constraint p02_brkga_comparar_deteccoes_pkey primary key (id),
  constraint p02_brkga_comparar_deteccoes_log_fkey foreign KEY (log_llm_id) references p02_brkga_logs_llm (id) on update CASCADE on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_brkga_comparar_deteccoes_log on public.p02_brkga_comparar_deteccoes_loop using btree (log_llm_id) TABLESPACE pg_default;

create index IF not exists idx_brkga_comparar_deteccoes_sessao on public.p02_brkga_comparar_deteccoes_loop using btree (sessao_brkga_id) TABLESPACE pg_default;