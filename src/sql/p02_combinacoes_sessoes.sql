create table public.p02_combinacoes_sessoes (
  combinacao_id bigint not null,
  sessao_id integer not null,
  constraint p02_combinacoes_sessoes_pkey primary key (combinacao_id, sessao_id),
  constraint p02_combinacoes_sessoes_combinacao_id_fkey foreign KEY (combinacao_id) references p02_combinacoes (id) on delete CASCADE,
  constraint p02_combinacoes_sessoes_sessao_id_fkey foreign KEY (sessao_id) references p02_sessoes (id) on delete RESTRICT
) TABLESPACE pg_default;