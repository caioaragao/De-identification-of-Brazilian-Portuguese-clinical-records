-- View: v_ranking_detectar_loop_primeiro
-- Fonte: p02_brkga_simular_injection (modo simulação com cascata completa).
-- Ranking do detector que disparou PRIMEIRO na cascata real (todos os 4
-- detectores juntos, parando no primeiro hit — critério cronológico). Em
-- desempate de mesmo chunk vale a ordem interna Det-1 -> Det-2 -> Det-3 -> Det-4
-- (sem bucket de empate: alimentar() devolve um único loop_tipo).
-- Escopo: sessão BRKGA 44 (via logs_llm -> resultados), apenas casos com loop
-- detectado (loop_detectado = true).
create or replace view public.v_ranking_detectar_loop_primeiro as
select
  p02_brkga_simular_injection.loop_tipo,
  count(*) as total_itens
from
  p02_brkga_simular_injection
  join p02_brkga_logs_llm on p02_brkga_simular_injection.log_llm_id = p02_brkga_logs_llm.id
  join p02_brkga_resultados on p02_brkga_logs_llm.resultado_id = p02_brkga_resultados.id
where
  p02_brkga_simular_injection.loop_detectado = true
  and p02_brkga_resultados.sessao_brkga_id = 44
group by
  p02_brkga_simular_injection.loop_tipo
order by
  (count(*));
