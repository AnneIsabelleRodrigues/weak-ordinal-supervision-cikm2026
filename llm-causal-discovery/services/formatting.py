import ast
import re
import json


def extract_list_from_text(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if not match:
        print("No list found in the text.")
        return None

    list_str = match.group(0)

    try:
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            print("The content found is not a list.")
            return None
    except Exception as e:
        print("Error interpreting the list:", e)
        try:
            list_str_fixed = re.sub(r"“|”", '"', list_str)
            return ast.literal_eval(list_str_fixed)
        except Exception as e2:
            print("It has not been possible to decode yet:", e2)
            return None


def extract_json_from_text(text):
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print("No JSON block found in the text.")
        return None

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        try:
            json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', json_str)
            return json.loads(json_str_fixed)
        except Exception as e2:
            print("It has not been possible to decode yet:", e2)
            return None


def parse_llm_answer(answer_content, valid_nodes, independent_nodes, current_node):
    if answer_content is None:
        print("WARNING: answer_content is None. Returning empty list.")
        return []

    try:
        answer_list = answer_content.split('<Answer>')[1].split('</Answer>')[0].split(',')
    except:
        print(f"WARNING: Could not find <Answer>...</Answer> tags. Attempting alternative analysis.")
        answer_list = []
        for node in valid_nodes:
            if node in answer_content and node != current_node:
                answer_list.append(node)

    cleaned_answer = []
    for node in answer_list:
        node = node.strip()

        if len(node) == 0 or node.lower() == "none":
            continue

        if node not in valid_nodes:
            continue

        if node in independent_nodes:
            continue

        if node == current_node:
            continue

        cleaned_answer.append(node)

    print(f"  Identified effects: {cleaned_answer if cleaned_answer else 'None'}")
    return cleaned_answer


def extract_variable_names(data: dict) -> list[str]:
    return [
        item["variable_name"]
        for category in data.values()
        for item in category
        if "variable_name" in item
    ]


def save_results(output_file, variables, adj_matrix, independent=None):
    data = {
        "pdf_name": output_file,
        "variables": variables,
        "adjacency_matrix": adj_matrix,
    }

    if independent is not None:
        data["independent_variables"] = independent

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_anchored_query(variables: dict, causal_graph: dict) -> str:

   clean_graph = {
       cause: list(dict.fromkeys(effects))
       for cause, effects in causal_graph.items()
   }

   var_lookup = {}
   for category, entries in variables.items():
       for entry in entries:
           name = entry.get("variable_name")
           if name:
               var_lookup[name] = {
                   "category": category,
                   "description": entry.get("description", ""),
                   "value": entry.get("value", ""),
                   "narrative_importance": entry.get("narrative_importance", "")
               }

   referenced = set()
   for cause, effects in clean_graph.items():
       referenced.add(cause)
       referenced.update(effects)

   category_order = [
       "patient_demographics",
       "clinical_characteristics",
       "laboratory_biomarkers",
       "treatment_data",
       "outcome_measures"
   ]

   by_category = {cat: [] for cat in category_order}
   uncategorized = []
   for name in sorted(referenced):
       if name in var_lookup:
           cat = var_lookup[name]["category"]
           if cat in by_category:
               by_category[cat].append(name)
           else:
               uncategorized.append(name)
       else:
           uncategorized.append(name)

   var_lines = ["{"]
   active_categories = [c for c in category_order if by_category[c]]
   if uncategorized:
       active_categories_with_extra = active_categories + ["uncategorized"]
   else:
       active_categories_with_extra = active_categories

   for ci, cat in enumerate(active_categories_with_extra):
       names = by_category.get(cat, uncategorized if cat == "uncategorized" else [])
       if not names:
           continue
       cat_comma = "," if ci < len(active_categories_with_extra) - 1 else ""
       var_lines.append(f'  "{cat}": [')
       for ni, name in enumerate(names):
           entry_comma = "," if ni < len(names) - 1 else ""
           if name in var_lookup:
               meta = var_lookup[name]
               var_lines.append(
                   f'    {{"variable_name": "{name}", '
                   f'"category": "{cat}", '
                   f'"description": "{meta["description"]}", '
                   f'"narrative_importance": "{meta["narrative_importance"]}"}}{entry_comma}'
               )
           else:
               var_lines.append(
                   f'    {{"variable_name": "{name}", '
                   f'"category": "", '
                   f'"description": "", '
                   f'"narrative_importance": ""}}{entry_comma}'
               )
       var_lines.append(f"  ]{cat_comma}")

   var_lines.append("}")
   var_block = "\n".join(var_lines)

   active_causes = {k: v for k, v in clean_graph.items() if v}
   skeleton_lines = ["{"]
   causes = list(active_causes.keys())

   for i, (cause, effects) in enumerate(active_causes.items()):
       cause_comma = "," if i < len(causes) - 1 else ""
       skeleton_lines.append(f'  "{cause}": [')
       for j, effect in enumerate(effects):
           effect_comma = "," if j < len(effects) - 1 else ""
           skeleton_lines.append(
               f'    {{"effect": "{effect}", '
               f'"plausibility": "FILL", '
               f'"confidence": "FILL", '
               f'"justification": "FILL"}}{effect_comma}'
           )
       skeleton_lines.append(f"  ]{cause_comma}")

   skeleton_lines.append("}")
   skeleton = "\n".join(skeleton_lines)


   return (
       f"VARIABLE DEFINITIONS:\n"
       f"Use definitions to understand variable meaning. Do NOT treat examples or narrative text as proof of causation.\n"
       f"{var_block}\n\n"
       f"TASK:\n"
       f"Complete the skeleton below. Rules:\n"
       f"- Replace each FILL in 'plausibility' with exactly one of: HIGH, MEDIUM, LOW\n"
       f"- Replace each FILL in 'confidence' with a float between 0.0 and 1.0\n"
       f"- Replace each FILL in 'justification' with one concise medical sentence\n"
       f"- Do NOT change variable names, keys, or structure\n"
       f"- Output only the completed JSON, nothing else\n\n"
       f"SKELETON:\n{skeleton}"
   )


def parse_causal_validation(raw_answer: str) -> dict:
    clean = (
        raw_answer.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e}\nRaw: {raw_answer[:300]}")

    for cause, effects in data.items():
        for entry in effects:
            confidence = entry.get("confidence")
            if confidence is None:
                raise ValueError(f"Missing 'confidence' in entry: {entry}")
            if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
                raise ValueError(f"Invalid confidence '{confidence}' for cause '{cause}'")
            entry["confidence"] = round(float(confidence), 3)

            if entry.get("plausibility") not in {"HIGH", "MEDIUM", "LOW"}:
                raise ValueError(f"Invalid plausibility in entry: {entry}")

            if "FILL" in str(entry.get("justification", "")):
                raise ValueError(f"Unfilled justification in entry: {entry}")

    return data