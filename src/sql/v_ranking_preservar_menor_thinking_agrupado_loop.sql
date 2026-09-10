-- View: v_ranking_preservar_menor_thinking_agrupado_loop
-- Fonte: p02_brkga_comparar_deteccoes_loop (modo offline --comparar-deteccoes).
-- Granularidade GROSSA: agrupa pelo vencedor por bucket (detector_mais_preciso),
-- com valores 'artigo' (Det-1/2/3), 'det4', 'empate' ou 'nenhum'. Vence quem
-- localizou o início do loop na MENOR posição, ou seja, quem cortou mais cedo e
-- preservou o menor thinking (removeu a maior parte do loop).
-- Escopo: sessão BRKGA 44, apenas logs com falha original (houve_falha = true).
create or replace view public.v_ranking_preservar_menor_thinking_agrupado_loop as
select
  p02_brkga_comparar_deteccoes_loop.detector_mais_preciso,
  count(*) as total_itens
from
  p02_brkga_comparar_deteccoes_loop,
  p02_brkga_logs_llm
where
  p02_brkga_comparar_deteccoes_loop.log_llm_id = p02_brkga_logs_llm.id
  and p02_brkga_logs_llm.houve_falha = true
  and p02_brkga_comparar_deteccoes_loop.sessao_brkga_id = 44
group by
  p02_brkga_comparar_deteccoes_loop.detector_mais_preciso
order by
  (count(*)) desc;
