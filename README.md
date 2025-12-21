**Purpose**
- Hybrid spaCy + LangGraph + deterministic compiler workflow that converts healthcare questions into validated GraphQL queries.
- Only [schema/schema.json](schema/schema.json) is domain-specific; all Python modules stay the same when the schema changes.

## Repository map

| Stage | Module(s) | Responsibility |
| --- | --- | --- |
| Mention extraction | [nlp/ner.py](nlp/ner.py) | spaCy entities with text/label/span metadata. |
| Agentic reasoning | [agentic/schema_reasoner.py](agentic/schema_reasoner.py), [agentic/ollama_client.py](agentic/ollama_client.py) | LangGraph loop that keeps querying Ollama (or mocks) until all filters are schema-valid. |
| Plan assembly | [pipeline/run.py](pipeline/run.py) | Orchestrates mentions, agent output, joins, and defaults before compiling. |
| Validation | [ir/logical_plan.py](ir/logical_plan.py), [ir/validator.py](ir/validator.py) | Ensures joins/filters/roots align with schema structure. |
| Compilation | [compiler/graphql_compiler.py](compiler/graphql_compiler.py) | Converts the logical plan into GraphQL. |
| Cypher conversion | [compiler/cypher_compiler.py](compiler/cypher_compiler.py) | Deterministic GraphQL → Cypher translator (schema-driven, no LLM). |
| Tests | [tests/run_checks.py](tests/run_checks.py), [tests/run_eval_30.py](tests/run_eval_30.py) | Regression harnesses (mocked + real models). |

## End-to-end flow

```mermaid
flowchart LR
	 A(User Question) --> B(spaCy NER\\n[nlp/ner.py])
	 B --> C(LangGraph Agent\\n[agentic/schema_reasoner.py])
	 C --> D(Logical Plan Builder\\n[pipeline/run.py])
	 D --> E(Validator\\n[ir/validator.py])
	 E --> F(GraphQL Compiler\\n[compiler/graphql_compiler.py])
	 F --> G(Output & Tests)
	 C -->|MOCK_OLLAMA=1| C1(Mock responses)
	 C -.feedback.-> C
```

## Stage breakdown

1. **Mention extraction – [nlp/ner.py](nlp/ner.py)**  
	Loads `en_core_web_trf` (falls back to `en_core_web_sm`) and returns `[{"text", "label", "span"}]`. These mentions are bundled into the LangGraph payload so the LLM sees contextual hints (cities, plan names, specialties) before generating filters.

2. **Agentic schema reasoner – [agentic/schema_reasoner.py](agentic/schema_reasoner.py)**  
	- Builds a prompt containing the question, schema JSON, auto-generated `Type.field` listing, spaCy mentions, and failure feedback.  
	- LangGraph nodes: `generate` (call [agentic/ollama_client.py](agentic/ollama_client.py)), `validate` (confirm each `field_path` exists), and a router that loops until filters are schema-valid or retries are exhausted.  
	- `_extract_json_block` and `force_retry` ensure any chatter is ignored; only clean JSON reaches downstream stages.

3. **Logical plan assembly – [pipeline/run.py](pipeline/run.py)**  
	- Loads the schema, merges agent filters, deduplicates them, and constructs [ir/logical_plan.LogicalPlan](ir/logical_plan.py).  
	- Default root is `Provider` when available; this is the only non-schema heuristic baked into the plan builder.

4. **Validation – [ir/validator.py](ir/validator.py)**  
	- Ensures every join/filter/select can be satisfied by the schema graph before compiling.  
	- Any bad filter that slips through raises `ValueError` with explicit attribute names.

5. **GraphQL compilation – [compiler/graphql_compiler.py](compiler/graphql_compiler.py)**  
	- Emits the final query without data-dependent logic; the backend can run it directly.

6. **Cypher conversion – [compiler/cypher_compiler.py](compiler/cypher_compiler.py)**  
	- Parses the deterministic GraphQL text and walks the schema relationships to emit Cypher using relationship labels such as `PROVIDER_AFFILIATIONS` — fully deterministic, no LLM calls.  
	- `process()` now returns both GraphQL and Cypher so downstream systems can choose either execution target.

7. **Testing & observability – [tests](tests)**  
	- `run_checks.py`: fast mocked regression to guard against collapsing outputs.  
	- `run_eval_30.py`: 30 real prompts to track end-to-end accuracy for any Ollama model (`phi3:14b`, `gpt-oss:20b`, `phi4:latest`, etc.).

## How spaCy NER constrains the LLM
- Mentions are passed to LangGraph inside the JSON payload (`payload["mentions"]`).  
- The prompt explicitly tells the model to respect those spans, e.g., if spaCy detects a city and a plan, those strings appear in the payload and typically reappear as `Facility.location.city` and `Facility.plansAccepted.name`.  
- You can further tighten constraints by pre-filtering the schema list the prompt displays (e.g., only show plan fields when plan-like entities are present).

## Running locally
```bash
cd /home/gpuuser0/gpuuser0_a/NMD/testing
python3 -m venv .venv && source .venv/bin/activate
pip install -U spacy langgraph langchain-core
python -m spacy download en_core_web_trf  # optional but recommended

# Fast regression with deterministic agent
MOCK_OLLAMA=1 python tests/run_checks.py

# Ad-hoc NL → GraphQL using mocks
MOCK_OLLAMA=1 PYTHONPATH=. python pipeline/run.py "Find active cardiology providers in Los Angeles hospitals that accept Blue Shield"

# Real agent run (requires Ollama CLI + model)
unset MOCK_OLLAMA
export OLLAMA_MODEL=phi3:14b  # gpt-oss:20b, phi4:latest, etc.
PYTHONPATH=. python pipeline/run.py "Show inactive oncology clinics in Seattle"
```

Each CLI run prints the GraphQL query followed by the deterministic Cypher translation for copy/paste into your target data store.

## Validation harnesses
- **Deterministic smoke test:** `MOCK_OLLAMA=1 python tests/run_checks.py`.  
- **Full evaluation:** `unset MOCK_OLLAMA && OLLAMA_MODEL=gpt-oss:20b PYTHONPATH=. python tests/run_eval_30.py` (scored **30/30** most recently). Swap `gpt-oss:20b` for any other local Ollama model when benchmarking.

## LLM fallback playbook
+ **JSON formatting issues:** LangGraph forces another iteration when non-JSON chatter appears. Increase `MAX_ATTEMPTS` or improve `_extract_json_block` if a model keeps ignoring instructions.
+ **Invalid attributes:** Errors such as `Rejected attributes not in schema` originate in [ir/validator.py](ir/validator.py). Update the schema (preferred) or adjust few-shot examples to steer the agent toward allowed fields.
+ **Agent exhaustion:** When `infer_filters` raises `ValueError` after max attempts, drop to mocks (`MOCK_OLLAMA=1`), provide a deterministic fallback filter set, or augment the prompt with additional hints.
+ **Runtime outages:** If Ollama is unavailable, set `MOCK_OLLAMA=1` to continue working locally and in CI.

## Domain-specific / hard-coded knobs
- `_MOCK_RESPONSES` in [agentic/ollama_client.py](agentic/ollama_client.py) encode sample filters (Blue Shield, Cigna, Austin). Update them when the domain changes.
- Few-shot examples in [agentic/schema_reasoner.py](agentic/schema_reasoner.py) mention specific specialties/cities/plans; edit these along with the schema.
- Scenario text in [tests/run_checks.py](tests/run_checks.py) and [tests/run_eval_30.py](tests/run_eval_30.py) is healthcare-specific by design.
- `MAX_ATTEMPTS = 3` and the default `Provider` root in [pipeline/run.py](pipeline/run.py) are heuristics that you can tune per schema.
- Everything else reads [schema/schema.json](schema/schema.json) at runtime; no silent hard-coding remains.

## Maintaining schema or prompt changes
1. Update [schema/schema.json](schema/schema.json) to reflect new entities/relations.  
2. Refresh few-shot examples and mock responses if new fields appear.  
3. Re-run `tests/run_checks.py` (mock) and `tests/run_eval_30.py` (real model).  
4. If repeated failures occur, add targeted feedback in `_generate_filters` or tighten how spaCy mentions filter the schema list.

By confining domain knowledge to the schema + prompt assets and using deterministic IR/validator/compiler components, the system stays portable across provider networks, clinical trials, or logistics graphs without touching Python code.
