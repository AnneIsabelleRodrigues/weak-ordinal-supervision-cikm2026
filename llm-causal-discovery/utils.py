import json
from pathlib import Path

from llm.pipeline import MedicalRAGPipeline
from services.dataset_loader import read_document
from services.data_pipeline import (
    variable_discovery_instruction,
    variable_initialization_instruction,
    llm_bfs_local,
    independent_validation_instruction,
    causal_validation_instruction,
)
from services.formatting import (
    extract_list_from_text,
    extract_json_from_text,
    extract_variable_names,
)
import uuid
import torch

def process_document(pdf_path: Path, model_name: str, role: str, feature_description: str, rag: MedicalRAGPipeline,
                     expertise: str) -> dict:

    article = read_document(pdf_path)

    # 1. Variable discovery
    variables = extract_json_from_text(
        variable_discovery_instruction(model_name, article, feature_description, role)
    )

    if variables is None:
        raise ValueError("Variable discovery returned no valid JSON.")

    # 2. Independent variable initialisation
    independent_vars = extract_list_from_text(
        variable_initialization_instruction(model_name, str(variables), role)
    )

    if independent_vars is None:
        raise ValueError("Independent variable extraction returned None.")

    # 3. RAG-based independence verification
    verification_raw = independent_validation_instruction(
        independent_vars,
        list(extract_variable_names(variables)),
        rag,
        expertise=expertise,
    )

    verified_independent = extract_list_from_text(verification_raw["answer"])
    if verified_independent is None:
        raise ValueError("Independence verification returned None.")

    # 4. Causal graph construction
    adj_matrix = llm_bfs_local(
        model_name,
        extract_variable_names(variables),
        verified_independent,
        variables,
        role=role,
    )
    if adj_matrix is None:
        raise ValueError("Adjacency matrix construction returned None.")

    return variables, adj_matrix, verified_independent


def load_persisted(persist_path: Path) -> tuple:
    with open(persist_path) as f:
        data = json.load(f)
    return data["variables"], data["adjacency_matrix"]


def validate_graph(variables, adj_matrix: dict, rag: MedicalRAGPipeline, role: str, expertise: str) -> str:
    result = causal_validation_instruction(
        variables,
        adj_matrix,
        rag,
        role=role,
        expertise=expertise,
    )
    print(result["answer"])
    return result["answer"]

def text_to_uuid(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()