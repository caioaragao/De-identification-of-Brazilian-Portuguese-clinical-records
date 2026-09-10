# Hybrid De-identification of Clinical Records with Local Language Models

A de-identification system for Brazilian Portuguese clinical records that combines
**statistical n-grams**, **regular expressions** and **inference by a local LLM served
through Ollama**, with incremental learning persisted in PostgreSQL and real-time
detection of reasoning loops.

---

## Why a hybrid pipeline

No single layer solves the problem on its own, and each one covers the failure mode of
the others:

| Layer | Covers | Fails on |
|---|---|---|
| **Statistical n-grams** | Standardized phrases recurring across patients (*boilerplate*) | Free, idiosyncratic text |
| **Regular expressions** | Rigid patterns: national ID numbers, dates, phone numbers, e-mails | Names, addresses, ambiguous terms |
| **Local LLM** | Semantic classification in context | Computational cost; hallucination; reasoning loops |

The n-gram phase runs **before** the LLM and strips from the corpus everything that is
provably safe, shrinking the text that reaches inference. What remains is sent to the
model line by line.

All processing is **offline** — no clinical data ever leaves the machine.

---

## Key features

**Cascaded detection of reasoning loops (`detector_loop.py`).** Models with
chain-of-thought fall into cycles that overflow the context window. Four detectors run as
a cascade during streaming, cheapest first, and each estimates the offset where the loop
began so the reasoning can be truncated:

| | Detector | Cost | Captures |
|---|---|---|---|
| Det-1 | Exact character window | O(window × threshold) | Short-period loops, including whitespace-free ones |
| Det-2 | Proportional n-gram repetition | O(n_tokens) | AAA and ABAB patterns, long cycles |
| Det-3 | Line-wise cosine similarity (TF-IDF) | O(window²) | Cycles with textual variation (paraphrase, reordering) |
| Det-4 | Repeated line blocks of variable period | O(N × stride) | Byte-identical multi-line blocks |

Det-2 and Det-3 adapt the *rep-n* and *statement-level* techniques of **DeRep**
(Liu et al., 2025); Det-1 is original, and Det-4 adapts the same work's *block-level*
detector, replacing TF-IDF similarity with exact equality. The substantive transposition
is from **offline post-processing over generated code** to **online detection over
streaming clinical text**, interrupting generation in real time.

**Dynamic Unicode tags.** Clinical records use `[` and `]` natively, which collides with
the masking syntax suggested to the model. For each session the system picks Unicode
characters absent from the corpus (e.g. `⦃PERSON 42⦄`) and binds them to the final
submission only.

**Resume capability.** Each session persists the learned entities and the index of the
last processed record. An interrupted run resumes exactly where it stopped; sessions
marked `concluido` are skipped before any write.

**Memoization across records.** PII learned in one record is applied in bulk to all others
during final consolidation — a name detected in record X is masked in Y and Z without a
new call to the LLM.

**Recovery from context window overflow.** When `done_reason != 'stop'`, the truncated
reasoning is injected into the next attempt's prompt, with `num_ctx` and `num_predict`
expanded and `think=False`. Governed by `PROMPT_INJECTION`.

**Partial age masking.** Ages are stored as the full expression (`"39 anos"`) but only the
numeral enters the mask, so that a temperature of `39°C` is not masked by collision — this
also matches the gold standard, which annotates the numeric token alone.

**Combinatorial reports with pruning.** Sessions are combined by union or intersection,
with Branch-and-Bound pruning on F2 and a `frozenset` cache (the same combination given in
a different argument order is not reprocessed).

---

## Layout

```
src/
├── main.py                  # CLI and orchestration of combinatorial reports
├── PipelineOrquestrador.py  # Pipeline: n-gram phase + line-by-line phase
├── Anonimizacao.py          # Masking engine, dynamic tags, PII maps
├── ClienteOllama.py         # LLM streaming, reasoning/answer split, retry
├── detector_loop.py         # Cascaded detection of reasoning loops
├── Supabase.py              # Relational persistence and state recovery
├── GeradorRelatorio.py      # Output reports
├── funcoes_gerais.py        # Tokenization, pre-processing, combinations
├── logger.py                # Level-based logging
├── a01_platform/            # Configuration profiles, selected by hostname
├── anonymed/                # Metrics, gold-standard normalization and corpus preparation
├── sql/                     # DDL for tables and views
└── tests/                   # pytest suite
```

---

## Requirements

- Python ≥ 3.12
- [Ollama](https://ollama.com) reachable (locally or over the network)
- A PostgreSQL/[Supabase](https://supabase.com) project with the tables in `src/sql/`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

**Credentials.** Never versioned. Copy the example and fill it in:

```bash
cp config/supabase.cfg.example config/supabase.cfg
```

**Environment profile.** `src/a01_platform/config_router.py` selects a `config.py` by the
machine's `hostname`, among the directories under `src/a01_platform/` (fallback: `vertex`).
For a new environment, create a directory named after your host using
`a01_platform/computador/config.py` as a template, add the name to `_PERFIS_VALIDOS`, and
set:

| Constant | What it defines |
|---|---|
| `PASTA_RAIZ` | Project root on disk |
| `SUPABASE_CFG_PATH` | Path to the credentials `.cfg` |
| `ARQUIVO_DS_ENTRADA` | Input dataset |
| `DIRETORIO_SAIDA` | Where masks and reports are written |
| `OLLAMA_URL`, `MODELOS_LLM` | Server and models to evaluate |
| `LLM_NUM_CTX`, `LLM_NUM_PREDICT` | Context window and generation ceiling |
| `PROMPT_INJECTION` | Recovery by reasoning injection (`True`) or hard failure (`False`) |
| `NGRAM_*`, `LOOP_*` | Hyperparameters of the n-gram phase and the four loop detectors |

## Usage

```bash
cd src

# Default run (uses the dataset and models of the active profile)
python main.py

# Overriding model, dataset and server
python main.py -m qwen3:14b \
               -f anonymed/datasets/clinical_deid_test_SEM_TAGS.json \
               --ollama-url http://localhost:11434 -l 100 -y

# Combinatorial report: union with F2 pruning, or intersection
python main.py -u 63 65 72 --grupo cpu
python main.py -i 63 65 72 --grupo cpu
```

<details>
<summary>All flags</summary>

| Flag | Effect |
|---|---|
| `-d/--database` | Path to the Supabase `.cfg` |
| `-f/--file_dataset` | Dataset (JSON or CSV) |
| `-u/--uniao IDs…` | Report by incremental union with F2 pruning |
| `-i/--interseccao IDs…` | Report by intersection |
| `--grupo NAME` | Label for the combination group (required with `-u`/`-i`) |
| `-l/--limit N` | Caps the number of records |
| `-m/--modelos …` | Overrides `MODELOS_LLM` |
| `--ollama-url URL` | Overrides `OLLAMA_URL` |
| `--log-level` | `silent \| error \| warning \| normal \| verbose` |
| `-g/--grafico [file]` | Writes a metrics chart as PNG |
| `-r/--report` | Keeps the physical artifacts in the output directory |
| `-y/--yes` | Skips the interactive confirmation |

</details>

## Tests

```bash
pytest
```

The `pythonpath` and `testpaths` settings in `pyproject.toml` are what let the suite run
from the repository root.

---

## Evaluation

Metrics are computed by `src/anonymed/gerar_metricas.py`, which compares the generated mask
against the gold standard **character-by-character, by position** rather than by category
label — making the result robust to divergences in tag naming. It reports **Recall,
Precision, F1 and F2** (F2 weights recall, which is what matters in de-identification:
letting a datum through is worse than over-masking).

Every gold-standard label consumes its region in the buffer, including labels of
subcategories outside the relevant set, so that correct detections adjacent to them are not
counted as false positives. Only contiguous mask groups of at least 3 characters count as a
false positive, filtering out punctuation noise between neighbouring labels.

Beyond the numbers, the evaluation writes visual buffers and `.fp.json` / `.fn.json`
dictionaries listing the terms actually missed or over-detected, for qualitative debugging.

### Dataset

This project is evaluated on the **AnonyMed-BR** corpus (clinical de-identification in
Brazilian Portuguese), which is **not redistributed here** — it is third-party data with its
own terms of use. Obtain it from the original publication:

> M. Schiezaro, G. Rosa, B. A. G. Campos and H. Pedrini. **Guardians of the data: NER and
> LLMs for effective medical record anonymization in Brazilian Portuguese**. *Frontiers in
> Public Health*, 13, 2026. <https://doi.org/10.3389/fpubh.2025.1717303>

The pipeline does not consume the corpus in its original form. Convert it first with
`src/anonymed/limpar_tags_dataset.py`, which turns inline tags
(`<SUBCATEGORY>…</SUBCATEGORY>`) into positional labels, corrects whitespace offsets and
discards records whose tag does not match the annotated position.

The script takes no arguments — it reads and writes fixed paths. Create the folder and drop
the three original partitions in it using exactly these names:

```
src/anonymed/datasets/
├── clinical_deid_training_set.json
├── clinical_deid_validation_set.json
└── clinical_deid_test_set.json
```

```bash
mkdir -p src/anonymed/datasets   # then copy the three files above
python src/anonymed/limpar_tags_dataset.py
```

It writes one `*_SEM_TAGS.json` per partition plus a concatenated
`clinical_deid_FULL_SEM_TAGS.json`. These are the files that `-f/--file_dataset` and the
`ARQUIVO_DS_ENTRADA` setting in `src/a01_platform/*/config.py` expect. A few tests exercise
the real corpus and are skipped automatically when it is absent.

There are **19 subcategories**, grouped into 8 parent categories:

| Category | Subcategories |
|---|---|
| NAME | `DOCTOR`, `PATIENT` |
| LOCATION | `CITY`, `COUNTRY`, `STATE`, `STREET`, `HOSPITAL`, `ORGANIZATION`, `LOCATION_OTHER` |
| CONTACT | `PHONE`, `EMAIL`, `ZIP` |
| ID | `IDNUM`, `MEDICAL_RECORD`, `HEALTH_PLAN` |
| DATE, AGE, PROFESSION, OTHER | — |

`OTHER` is deliberately included in the computation so that results remain directly
comparable to the figures published by the corpus's original author.

> The data are **synthetic surrogates** — names, national ID numbers, phone numbers and
> e-mail addresses were generated, belong to no real person and correspond to no patient.

---

## Database schema

| Table | Contents |
|---|---|
| `p02_sessoes` | Sessions: model, status, tag characters, progress |
| `p02_conhecimento` | Entities learned per session (idempotent upsert) |
| `p02_logs_execucao` | Per-inference log: original text, pre-LLM text, raw answer, Ollama metrics |
| `p02_combinacoes` / `p02_combinacoes_sessoes` | Branch-and-Bound combinations and N-N pivot |
| `p02_brkga_*` | Sessions, results and logs of the hyperparameter tuning module |

Full DDL in [`src/sql/`](src/sql/).

---

## Citing

If this work is useful in your research, please cite the preprint:

> ARAGAO, C. G.; OLIVEIRA, M. C.; VIEIRA, T. M. A. **De-identification of Brazilian
> Portuguese Clinical Records Using a Hybrid Pipeline with Local Language Models**.
> Preprint, 2026. Available at:
> <https://www.researchgate.net/publication/411140278_De-identification_of_Brazilian_Portuguese_Clinical_Records_Using_a_Hybrid_Pipeline_with_Local_Language_Models>

```bibtex
@misc{aragao2026deidentification,
  author       = {Arag{\~a}o, Caio Galv{\~a}o and
                  Oliveira, Marcelo Costa and
                  Vieira, Thales Miranda de Almeida},
  title        = {De-identification of {Brazilian} {Portuguese} Clinical Records
                  Using a Hybrid Pipeline with Local Language Models},
  year         = {2026},
  howpublished = {Preprint},
  url          = {https://www.researchgate.net/publication/411140278_De-identification_of_Brazilian_Portuguese_Clinical_Records_Using_a_Hybrid_Pipeline_with_Local_Language_Models}
}
```

The corresponding dissertation (PPGI/UFAL) has not been deposited yet; this section will be
updated once a definitive reference exists.

---

## Reference

M. Liu, J. Li, Y. Wang, X. Du, Z. Ou, Q. Chen, B. An, Z. Wei, Y. Xu, F. Zou, X. Peng and
Y. Lou. **Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation**.
arXiv:2504.12608 [cs.SE], 2025. <https://arxiv.org/abs/2504.12608>

Conceptual basis for detectors Det-2, Det-3 and Det-4.

---

## License

Distributed under the [MIT License](LICENSE), which covers the code in this repository.

No third-party data is redistributed here. The AnonyMed-BR corpus used for evaluation must
be obtained from its original source and remains subject to its own terms of use — see
[Dataset](#dataset).
