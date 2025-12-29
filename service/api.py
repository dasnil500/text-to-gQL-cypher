"""FastAPI service exposing NL question to GraphQL/Cypher endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.run import process as run_pipeline


DEFAULT_SCHEMA = Path("schema/schema.json").as_posix()
app = FastAPI(
    title="text-to-gQL-cypher API",
    description="Convert natural-language healthcare queries into GraphQL or Cypher.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    schema_path: Optional[str] = None


class GraphQLResponse(BaseModel):
    graphql: str


class CypherResponse(BaseModel):
    cypher: str


def _resolve_schema_path(requested: Optional[str]) -> str:
    if requested:
        return requested
    return DEFAULT_SCHEMA


def _run_question(question: str, schema_path: Optional[str]):
    try:
        return run_pipeline(question, schema_path=_resolve_schema_path(schema_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - wide net for service stability
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/graphql", response_model=GraphQLResponse)
async def graphql_endpoint(payload: QueryRequest) -> GraphQLResponse:
    """Return only the GraphQL translation for a natural-language question."""
    result = _run_question(payload.question, payload.schema_path)
    return GraphQLResponse(graphql=result["query"].strip())


@app.post("/cypher", response_model=CypherResponse)
async def cypher_endpoint(payload: QueryRequest) -> CypherResponse:
    """Return only the Cypher translation for a natural-language question."""
    result = _run_question(payload.question, payload.schema_path)
    return CypherResponse(cypher=result["cypher"].strip())


@app.post("/both")
async def combined_endpoint(payload: QueryRequest):
    """Return both GraphQL and Cypher for convenience when callers need both."""
    result = _run_question(payload.question, payload.schema_path)
    return {
        "graphql": result["query"].strip(),
        "cypher": result["cypher"].strip(),
        "plan": {
            "root": result["plan"].root,
            "select": result["plan"].select,
        },
    }
