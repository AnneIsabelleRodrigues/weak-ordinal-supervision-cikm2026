import json
import argparse
from pathlib import Path

from config.env import CASE_REPORTS_DIR, OUTPUT_FILE_DIR
from config.settings import RAGConfig
from constants import VALID_MODELS, FEATURES_SELECTION, ROLES_SELECTION
from llm.pipeline import MedicalRAGPipeline
from services.formatting import (
    save_results,
)

from utils import (
    process_document,
    load_persisted,
    validate_graph
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=VALID_MODELS)
    parser.add_argument("--benchmark",  required=True, choices=list(FEATURES_SELECTION))
    return parser.parse_args()


def main():
    args = parse_args()
    model_name, benchmark = args.model_name, args.benchmark

    print(f"Model: {model_name}  |  Benchmark: {benchmark}")

    feature_description = FEATURES_SELECTION[benchmark]
    roles = ROLES_SELECTION[benchmark]
    role, expertise = roles["role"], roles["expertise"]

    rag = MedicalRAGPipeline(RAGConfig())

    case_dir = Path(CASE_REPORTS_DIR) / benchmark
    partial_path = Path(OUTPUT_FILE_DIR) / benchmark / "verifications_partial.json"
    final_path = Path(OUTPUT_FILE_DIR) / benchmark / "verifications.json"

    pdf_files = [f for f in case_dir.iterdir() if f.suffix.lower() in {".md", ".txt"}]

    results = []

    for pdf_path in pdf_files:
        print(f"\n===== PROCESSANDO {pdf_path.name} =====\n")
        persist_path = Path(OUTPUT_FILE_DIR) / benchmark / f"persist_{pdf_path.stem}.json"

        try:
            if persist_path.exists():
                print("Persisted file found.")
                variables, adj_matrix = load_persisted(persist_path)
            else:
                variables, adj_matrix, verified_independent = process_document(
                    pdf_path, model_name, role, feature_description, rag, expertise
                )
                save_results(str(persist_path), variables, adj_matrix, verified_independent)

            answer = validate_graph(variables, adj_matrix, rag, role, expertise)
            results.append(answer)

            partial_path.write_text(
                json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8"
            )

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[{pdf_path.name}] Skipping due to error: {e}")

    final_path.write_text(
        json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()