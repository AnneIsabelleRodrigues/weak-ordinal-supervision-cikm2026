import os

from services.api_call import ask_llm
from services.formatting import parse_llm_answer, build_anchored_query

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ============================================================
# BUILDER
# ============================================================

def variable_initialization_instruction(provider, variables, role="a pediatric cardiologist"):
    system_instruction = (
        f"You are a structured causal reasoning assistant supporting {role}.\n"
        "You are given a fixed list of variables related to a disease.\n\n"

        "DEFINITIONS:\n"
        "- A variable A causes B if A is explicitly or strongly implied to precede and influence directly B.\n"
        "- A variable is considered a ROOT variable if NO other variable in the list causes it.\n\n"

        "RULES:\n"
        "- Consider ONLY relationships between variables in the provided list.\n"
        "- DO NOT introduce external variables.\n"
        "- Prefer conservative reasoning: only assume causality if it is clear and well-established.\n"
        "- If causality is uncertain, assume NO causal link.\n"
        "- DO NOT use vague assumptions or speculative reasoning.\n\n"

        f"VARIABLE LIST:\n{variables}\n"
    )

    query_text = (
        "Task:\n"
        "1. Evaluate all possible pairwise causal relationships between variables.\n"
        "2. Determine which variables have NO incoming causal relationships from any other variable in the list.\n"
        "3. These variables are considered ROOT variables.\n\n"

        "OUTPUT REQUIREMENTS:\n"
        "- Return ONLY a valid Python list.\n"
        "- Each element must be an EXACT match to a variable name from the list.\n"
        "- Output MUST start with '[' and end with ']'.\n"
        "- Do NOT include explanations, comments, or extra text.\n"
        "- If no root variables exist, return an empty list [].\n"
    )

    try:
        return ask_llm(provider=provider, prompt=query_text, system_instruction=system_instruction)
    except Exception as e:
        print(f"[variable_initialization_instruction] Error: {e}")
        raise


def variable_discovery_instruction(provider, new_article, feature_descriptions, role="a pediatric cardiologist"):
    known_vars_list = "\n".join([
        f"- {name}: {desc}" for name, desc in feature_descriptions.items()
    ])

    system_instruction = (
        f"You are a highly precise information extraction system assisting {role}.\n"
        "Your task is to extract structured variables strictly from a scientific case report.\n\n"

        "GENERAL RULES:\n"
        "- Extract ONLY information explicitly stated in the text.\n"
        "- DO NOT infer, assume, or use external knowledge.\n"
        "- If a value is not explicitly mentioned, return null.\n"
        "- DO NOT create new variables.\n"
        "- ONLY use variable names from the provided list.\n"
        "- If no variables are found for a category, return an empty list [].\n\n"

        "VARIABLE EXTRACTION:\n"
        "For each identified variable, return:\n"
        "- variable_name: must EXACTLY match one of the known variables.\n"
        "- description: short, objective definition (max 15 words).\n"
        "- value: exact value(s) from the text.\n"
        "- narrative_importance: short explanation (max 30 words) describing:\n"
        "  * its role in the case\n"
        "  * any explicitly stated clinical or causal relevance\n"
        "- If multiple values exist, separate them using semicolons (;).\n\n"

        "KNOWN VARIABLES:\n"
        f"{known_vars_list}\n\n"

        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object with the following schema:\n\n"
        "{\n"
        '  "patient_demographics": [\n'
        '    {"variable_name": "...", "description": "...", "value": "...", "narrative_importance": "..."}\n'
        '  ],\n'
        '  "clinical_characteristics": [\n'
        '    {"variable_name": "...", "description": "...", "value": "...", "narrative_importance": "..."}\n'
        '  ],\n'
        '  "treatment_data": [\n'
        '    {"variable_name": "...", "description": "...", "value": "...", "narrative_importance": "..."}\n'
        '  ],\n'
        '  "outcome_measures": [\n'
        '    {"variable_name": "...", "description": "...", "value": "...", "narrative_importance": "..."}\n'
        '  ],\n'
        '  "laboratory_biomarkers": [\n'
        '    {"variable_name": "...", "description": "...", "value": "...", "narrative_importance": "..."}\n'
        '  ]\n'
        "}\n\n"

        "STRICT OUTPUT CONSTRAINTS:\n"
        "- Output MUST start with '{' and end with '}'.\n"
        "- DO NOT include explanations, notes, or text outside JSON.\n"
        "- Ensure valid JSON syntax (double quotes, commas, brackets).\n"
        "- DO NOT omit any required keys.\n"
    )

    query_text = (
        "[ARTICLE TO ANALYZE]\n"
        f"{new_article}\n\n"
        "[MAIN TASK]\n"
        "Extract all variables from the article and return them in JSON format.\n"
    )

    try:
        return ask_llm(provider=provider, prompt=query_text, system_instruction=system_instruction)
    except Exception as e:
        print(f"[variable_discovery_instruction] Error: {e}")
        raise


# ============================================================
# BFS
# ============================================================

def llm_bfs_local(provider, var_names_and_desc, independent_nodes, variablesdiscovered, role):
    try:
        print("=" * 60)
        print("INICIANDO LLM-BFS LOCAL")
        print("=" * 60)

        prompt_format = (
            "OUTPUT FORMAT (JSON only):\n\n"
            "{\n"
            '  "effects": ["var1", "var2"]\n'
            "}\n\n"

            "RULES:\n"
            "- Include ONLY variables directly caused by the target variable.\n"
            "- Do NOT include indirect effects.\n"
            "- Do NOT include the target variable itself.\n"
            "- Do NOT repeat variables.\n"
            "- Use exact variable names as provided.\n"
            "- If no direct effects exist, return:\n"
            '{"effects": []}\n'
            "- Output MUST be valid JSON.\n"
            "- No explanations, no extra text.\n"
        )

        system_instruction = (
            f"You are a causal reasoning assistant helping {role}. "
            "You analyze causal relationships between biomedical variables.\n\n"

            "OBJECTIVE:\n"
            "Identify plausible direct causal effects of a given variable.\n\n"

            "DEFINITIONS:\n"
            "- A variable A causes B if A is explicitly or strongly implied to precede and influence directly B.\n"

            "INCLUSION POLICY:\n"
            "- Use a LIBERAL inclusion threshold: include a relationship if there is reasonable "
            "  mechanistic evidence, even if the link is not universally established.\n"
            "- When uncertain whether to include a variable, INCLUDE IT.\n"
            "- Include relationships that are indirect in general biology but become direct "
            "  within the scope of the provided variable set.\n"
            "- Only exclude a relationship if there is no plausible mechanistic pathway "
            "  or if the direction is clearly reversed.\n\n"
            "  Pay extra attention to the temporality of relationships. The response or overall status should be interpreted as an effect of preceding factors, not as a cause.\n\n"

            "RULES:\n"
            "- Consider ONLY variables provided.\n"
            "- Do NOT introduce new variables.\n"
            "- Only include relationships where the cause temporally precedes and mechanistically influences the effect.\n"
            "- Do NOT include reverse or bidirectional relationships.\n"

            f"VARIABLES:\n{variablesdiscovered}\n"
        )

        nodes = list(var_names_and_desc)
        unvisited_nodes = set(nodes)
        for indep in independent_nodes:
            if indep in unvisited_nodes:
                unvisited_nodes.remove(indep)

        predict_graph = {}
        frontier = list(independent_nodes)

        level = 0
        while frontier:
            level += 1
            next_frontier = []

            for current_node in frontier:
                prompt = (
                    f"Given {', '.join(independent_nodes)} as root causes not affected by any variable, "
                    f"and the causal relationships already found:\n"
                )

                for head, tails in predict_graph.items():
                    if tails:
                        prompt += f"{head} causes {', '.join(tails)}.\n"

                prompt += (
                    f"\nNow, analyze which of the following variables {nodes} are directly caused by {current_node}. "
                    f"For each link {current_node} → X, explain briefly the reasoning or mechanism. "
                )

                query_text = prompt + "\n\n" + prompt_format

                try:
                    answer_content = ask_llm(provider=provider, prompt=query_text,
                                             system_instruction=system_instruction)
                except Exception as e:
                    print(f"[llm_bfs_local] Error querying node '{current_node}': {e}")
                    predict_graph[current_node] = []
                    continue

                try:
                    cleaned_answer = parse_llm_answer(answer_content, nodes, independent_nodes, current_node)
                except Exception as e:
                    print(f"[llm_bfs_local] Error parsing answer for node '{current_node}': {e}")
                    cleaned_answer = []

                predict_graph[current_node] = cleaned_answer

                for node in cleaned_answer:
                    if node in unvisited_nodes:
                        unvisited_nodes.remove(node)
                        next_frontier.append(node)

            if not next_frontier and unvisited_nodes:
                new_indep = list(unvisited_nodes)
                next_frontier.extend(new_indep)
                for node in new_indep:
                    unvisited_nodes.remove(node)

            frontier = next_frontier

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        for head, tails in predict_graph.items():
            if tails:
                print(f"{head} → {', '.join(tails)}")

        return predict_graph

    except Exception as e:
        print(f"[llm_bfs_local] Fatal error: {e}")
        raise


# ============================================================
# VALIDATION
# ============================================================

def causal_validation_instruction(variables, causal_graph, rag_prod, role, expertise):
    system_instruction = (
    f"You are a Senior Expert in Causal Inference and Clinical Medicine assisting {role}.\n\n"
    
    "TASK:\n"
    "You will receive a JSON skeleton with medical variable relationships.\n"
    "Your job is to fill in ONLY the three placeholder fields per entry:\n"
    "  - 'plausibility': replace FILL with exactly one of: HIGH, MEDIUM, LOW\n"
    "  - 'confidence': replace FILL with a float between 0.0 and 1.0\n"
    "  - 'justification': replace FILL with one concise medical sentence (max 20 words)\n\n"
    
    "RULES:\n"
    "1. Return ONLY the completed JSON.\n"
    "2. Do NOT change variable names, keys, brackets, or JSON structure.\n"
    "3. Use only plain ASCII characters in your text. No Unicode, no escaped chars.\n"
    "4. Every FILL must be replaced. No FILL should remain in the output.\n\n"
    
    "CAUSAL REASONING RULES:\n"
    "1. Evaluate DIRECT causation only.\n"
    "   A causes B only if changing A would directly change B through a known mechanism.\n\n"
    "2. Do NOT conflate correlation with causation.\n"
    "   If A and B share a common upstream cause, A does not cause B.\n\n"
    "3. Do NOT confuse diagnostic workflow with causation.\n"
    "   A causes B' must reflect a physiological effect, not a clinical decision triggered by A.\n"
    "4. Respect causal direction strictly.\n\n"
    "   The skeleton defines A → B. Evaluate only whether A directly causes B.\n"
    " Do not reason about B → A or treat bidirectional relationships as unidirectional.\n\n"
    "5. Observation and report variables are always HIGH plausibility from their direct source.\n"
    "   If A is directly measured or documented as B, the link is HIGH.\n"
    "   Justify with the act of measurement or documentation, not with physiology.\n\n"
    
    
    "VARIABLE DEFINITIONS:\n"
    "Each variable may include metadata. Use metadata only to understand semantics of each variable.:\n"
    "- variable_name: the variable described\n"
    "- category: if the variable is categorized as patient demographics, clinical characteristics, laboratory biomarkers or outcome measures\n"
    "- description: clinical meaning\n"
    "- narrative_importance: context in the article\n"
    "Use this information only to interpret the variable semantics, not as proof of causation.\n\n"
    "The proposed edges may be incorrect. Evaluate independently and critically.\n\n"
    "The categories might indicate likely causal roles: demographics = upstream, clinical states = intermediate, treatments = interventions, outcomes = downstream.\n\n"
    
    "PLAUSIBILITY GUIDE:\n"
    " HIGH - well-established direct causal mechanism in clinical literature\n"
    " MEDIUM - plausible but indirect, context-dependent, or debated\n"
    " LOW - weak, speculative, or no established causal pathway\n\n"
    
    "CONFIDENCE GUIDE:\n"
    "  Confidence MUST be a decimal number using only ASCII digits 0-9 and period.\n"
    "  0.9-1.0 : well-established, strong and consistent clinical evidence\n"
    "  0.7-0.89: plausible, commonly accepted, moderate evidence\n"
    "  0.5-0.69: possible but uncertain, context-dependent, or debated\n"
    "  0.3-0.49: weak or speculative association\n"
    "  0.0-0.29: contradicted or no established causal pathway\n"
    " Use higher values only when the relationship is well-established.\n\n"
    
    "EXAMPLE INPUT SKELETON:\n"
    '{\n'
    '  "Infection": [\n'
    '    {"effect": "Fever", "plausibility": "FILL", "confidence": "FILL", "justification": "FILL"},\n'
    '    {"effect": "Sick", "plausibility": "FILL", "confidence": "FILL", "justification": "FILL"}\n'
    '  ]\n'
    '}\n\n'
    
    "EXAMPLE OUTPUT:\n"
    '{"Infection": [{"effect": "Fever", "plausibility": "HIGH", "confidence": 0.92, "justification": "Pathogens trigger cytokine release raising the hypothalamic set point."}, {"effect": "Sick", "plausibility": "HIGH", "confidence": 0.88, "justification": "Infection impairs normal physiological function causing systemic illness."}]}\n\n'
    
    "ABSOLUTE RULE: Output EXACTLY the same keys and effects as the skeleton. "
    )

    query = build_anchored_query(variables, causal_graph)

    print(query)

    try:
        return rag_prod.query(system_instruction, query)
    except Exception as e:
        print(f"[causal_validation_instruction] Error: {e}")
        raise


def independent_validation_instruction(selected_variables, all_variables, rag_prod, expertise):
    system_instruction = (
        f"You are a causal reasoning assistant with expertise in {expertise}.\n\n"

        "OBJECTIVE:\n"
        "Validate whether a subset of variables is truly causally independent\n"
        "with respect to a larger set of variables.\n\n"

        "DEFINITIONS:\n"
        "- A variable A causes B if A biologically or clinically precedes and influences B.\n"
        "- A variable is NOT independent if it is caused by ANY other variable in the full variable set.\n\n"

        "RULES:\n"
        "- Consider causal relationships ONLY among the provided variables.\n"
        "- Use both the full variable list and the selected subset.\n"
        "- Do NOT introduce external variables.\n"
        "- Use conservative reasoning:\n"
        "  - Only assign causality if it is clear and well-established.\n"
        "  - If uncertain, assume NO causal relationship.\n\n"

        "TASK:\n"
        "- For each variable in the selected subset, check if it is caused by any variable in the full list.\n"
        "- Remove variables that have any plausible incoming causal relationship.\n"
        "- Keep only variables that remain causally independent.\n"
    )

    query = (
        "FULL VARIABLE LIST:\n"
        f"{all_variables}\n\n"

        "SELECTED VARIABLES (assumed independent):\n"
        f"{selected_variables}\n\n"

        "OUTPUT:\n"
        "Return a plain text list containing the REVISED set of selected (independent) variables.\n\n"

        "RULES:\n"
        "- Remove variables from the selected variables that are causally dependent on any variable in the full variable list.\n"
        "- Do NOT add new variables.\n"
        "- Each variable must appear at most once.\n"
        "- Output must start with '[' and end with ']'.\n"
        "- No explanations, no comments, no extra text.\n\n"
        "- The majority of variables will not be independent.\n\n"

        "Example:\n"
        "[\"VarA\", \"VarC\"]\n"
    )

    try:
        return rag_prod.query(system_instruction=system_instruction, query=query)
    except Exception as e:
        print(f"[independent_validation_instruction] Error: {e}")
        raise