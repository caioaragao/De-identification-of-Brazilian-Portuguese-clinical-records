create table public.p02_combinacoes (
  id bigint generated always as identity not null,
  modo character varying(15) not null,
  nivel_k smallint not null,
  label_combo text not null,
  nome_arquivo text null,
  precision numeric(6, 4) null,
  recall numeric(6, 4) null,
  f1_score numeric(6, 4) null,
  f2_score numeric(6, 4) null,
  podado boolean not null default false,
  created_at timestamp with time zone not null default now(),
  constraint p02_combinacoes_pkey primary key (id)
) TABLESPACE pg_default;

create unique INDEX IF not exists uidx_combinacoes_label on public.p02_combinacoes using btree (label_combo) TABLESPACE pg_default;

create index IF not exists idx_combinacoes_modo_nivel on public.p02_combinacoes using btree (modo, nivel_k, f2_score desc) TABLESPACE pg_default;