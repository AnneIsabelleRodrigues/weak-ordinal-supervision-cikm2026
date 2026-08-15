import json
import re
from pathlib import Path
import gzip
import shutil

import networkx as nx
from pgmpy.readwrite import BIFReader
import tempfile


def _unescape_gemini(text: str) -> str:
    text = text.replace('\\\\"', '"')
    text = text.replace("\\\\'", "'")
    text = text.replace('\\\\', '')
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    return text


def _fix_encoding_artifacts(text: str) -> str:
    return text.replace('檛', 't')


def _remove_invalid_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\\/bfnrt]|u[0-9a-fA-F]{4})', '', text)


def _fix_inner_quotes(text: str) -> str:
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            result.append(c)
            result.append(text[i + 1])
            i += 2
            continue
        if c == '"':
            if not in_string:
                in_string = True
                result.append(c)
            else:
                j = i + 1
                while j < len(text) and text[j] == ' ':
                    j += 1
                if j < len(text) and text[j] in (':', ',', '}', ']', '\n'):
                    in_string = False
                    result.append(c)
                else:
                    result.append(' ')
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _close_json(text: str) -> str:
    open_arrays = max(0, text.count('[') - text.count(']'))
    open_objects = max(0, text.count('{') - text.count('}'))
    return text + ']' * open_arrays + '}' * open_objects


def parse_entry(entry: str) -> dict:
    if not entry or not entry.strip().startswith('`') and len(entry) < 50:
        return {}

    text = re.sub(r'^```\w*\n', '', entry)
    text = re.sub(r'\n```\s*$', '', text).strip()

    if text.startswith('"'):
        text = text[1:]

    text = _unescape_gemini(text)
    text = _fix_encoding_artifacts(text)
    text = _remove_invalid_backslashes(text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = _fix_inner_quotes(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    closed = _close_json(text)
    try:
        return json.loads(closed)
    except json.JSONDecodeError:
        pass

    for i in range(len(text) - 1, 0, -1):
        if text[i] in ('}', ']'):
            candidate = _close_json(text[: i + 1])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return {}


def parse_causal_verifications(filepath: str | Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        raw_list = json.load(f)
    return [parse_entry(entry) for entry in raw_list]


def read_groundtruth_graph(filepath: str | Path):

    filepath = Path(filepath)

    with tempfile.NamedTemporaryFile(
        suffix=".bif",
        delete=False
    ) as tmp_file:

        tmp_bif = Path(tmp_file.name)

    try:

        with gzip.open(filepath, "rb") as f_in:
            with open(tmp_bif, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        reader = BIFReader(str(tmp_bif))
        model = reader.get_model()

        missing_cpds = []

        for node in model.nodes():
            if model.get_cpds(node) is None:
                missing_cpds.append(node)

        if missing_cpds:
            raise ValueError(
                f"CPDs ausentes no modelo: {missing_cpds}"
            )

        return model

    finally:

        if tmp_bif.exists():
            tmp_bif.unlink()


def build_dag_true(model) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(list(model.nodes()))
    dag.add_edges_from(list(model.edges()))
    return dag


def normalize(text: str) -> str:
    return text.strip()


def filter_and_map(data: list, allowed_vars) -> list:
    allowed = set(allowed_vars)
    result = []

    for d in data:
        new_dict = {}

        for key, effects in d.items():
            key_norm = normalize(key)

            if key_norm not in allowed:
                continue

            filtered_effects = []
            for e in effects:
                effect_norm = normalize(e["effect"])

                if effect_norm in allowed:
                    e_copy = e.copy()
                    e_copy["effect"] = effect_norm
                    filtered_effects.append(e_copy)

            if filtered_effects:
                new_dict[key_norm] = filtered_effects

        if new_dict:
            result.append(new_dict)

    return result