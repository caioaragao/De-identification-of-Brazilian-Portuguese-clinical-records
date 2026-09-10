-- View: v_ranking_preservar_menor_thinking_loop
-- Fonte: p02_brkga_comparar_deteccoes_loop (modo offline --comparar-deteccoes).
-- Granularidade FINA: mesma população da versão agrupada, porém "explode" o
-- bucket 'artigo' no detector real (artigo_tipo = Det-1/2/3) e resolve 'det4' em
-- det4_tipo; 'empate' vira o rótulo combinado "Empate (artigo_tipo + det4_tipo)".
-- Ranking 4-way de quem cortou mais cedo (menor posição = menor thinking
-- preservado), usado na tabela de técnica vencedora.
-- Escopo: sessão BRKGA 44, apenas logs com falha original (houve_falha = true).
create or replace view public.v_ranking_preservar_menor_thinking_loop as
select
  case
    when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'artigo'::text then p02_brkga_comparar_deteccoes_loop.artigo_tipo
    when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'det4'::text then p02_brkga_comparar_deteccoes_loop.det4_tipo
    when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'empate'::text then (
      (
        (
          (
            'Empate ('::text || COALESCE(
              p02_brkga_comparar_deteccoes_loop.artigo_tipo,
              'NULL'::character varying
            )::text
          ) || ' + '::text
        ) || COALESCE(
          p02_brkga_comparar_deteccoes_loop.det4_tipo,
          'NULL'::character varying
        )::text
      ) || ')'::text
    )::character varying
    else p02_brkga_comparar_deteccoes_loop.detector_mais_preciso
  end as tecnica_vencedora,
  count(*) as total_itens
from
  p02_brkga_comparar_deteccoes_loop
  join p02_brkga_logs_llm on p02_brkga_comparar_deteccoes_loop.log_llm_id = p02_brkga_logs_llm.id
where
  p02_brkga_logs_llm.houve_falha = true
  and p02_brkga_comparar_deteccoes_loop.sessao_brkga_id = 44
group by
  (
    case
      when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'artigo'::text then p02_brkga_comparar_deteccoes_loop.artigo_tipo
      when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'det4'::text then p02_brkga_comparar_deteccoes_loop.det4_tipo
      when p02_brkga_comparar_deteccoes_loop.detector_mais_preciso::text = 'empate'::text then (
        (
          (
            (
              'Empate ('::text || COALESCE(
                p02_brkga_comparar_deteccoes_loop.artigo_tipo,
                'NULL'::character varying
              )::text
            ) || ' + '::text
          ) || COALESCE(
            p02_brkga_comparar_deteccoes_loop.det4_tipo,
            'NULL'::character varying
          )::text
        ) || ')'::text
      )::character varying
      else p02_brkga_comparar_deteccoes_loop.detector_mais_preciso
    end
  )
order by
  (count(*)) desc;
