# -*- coding: utf-8 -*-
"""
End-to-end dataset builder with label-aware, non-leaking context for Math-BERT.
"""

import os, re, time, json, random
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

# ----------------------------
# Optional: OpenAI client setup
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_HAS_OPENAI = False
try:
    from openai import OpenAI
    _client = OpenAI()
    # Simple ping: if no key, this will fail later and we'll fall back to dry_run
    _HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _HAS_OPENAI = False

# ----------------------------
# Basic text helpers
# ----------------------------

def extract_problem_answer(text: str) -> Tuple[str, str]:
    """
    Expected format: "Problem: <...> | Student Answer: <...>"
    Returns (problem, student_answer) or ("","") if not found.
    """
    if not text:
        return "", ""
    m = re.search(r"Problem:\s*(.*?)\s*\|\s*Student Answer:\s*(.*)\s*$", str(text))
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()

def text_to_latex(s: str) -> str:
    """A minimal LaTeXifier for fractions and x/÷ symbols; keep it simple."""
    s = s.replace("×", "x").replace("÷", "÷")
    # \frac{a}{b}
    def _frac(m):
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    s = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", _frac, s)
    s = s.replace(" x ", " \\times ").replace(" * ", " \\times ").replace(" ÷ ", " \\div ").replace("-", " - ")
    return s

def text_to_opt(s: str) -> str:
    """
    Very light tokenization for your OPT-like representation.
    FRAC a b, operators: ADD/SUB/MUL/DIV, mixed numbers -> "W FRAC a b"
    """
    tokens: List[str] = []

    # Mixed number: "w a/b"
    def _mixed_repl(m):
        w, a, b = m.group(1), m.group(2), m.group(3)
        tokens.extend([w, "FRAC", a, b])
        return ""

    s_work = " " + s + " "
    s_work = re.sub(r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\b", _mixed_repl, s_work)

    # Fractions:
    for a, b in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", s_work):
        tokens.extend(["FRAC", a, b])

    # Operators:
    if re.search(r"(?:\bx\b|\*|×)", s):
        tokens.insert(0, "MUL")
    elif "+" in s:
        tokens.insert(0, "ADD")
    elif re.search(r"(?<!\w)-", s):
        tokens.insert(0, "SUB")
    elif "÷" in s or "/" in s and " " in s:  # crude fallback for division symbol in problem text
        tokens.insert(0, "DIV")

    return " ".join(tokens) if tokens else ""

# ----------------------------
# Label → family/subtype helpers
# ----------------------------

def _family_from_label(label: str) -> str:
    if label.startswith("conv-"):
        return "conversion"
    if label.startswith("po-"):
        return "property_of_operation"
    if label.startswith("pf-"):
        return "property_of_fractions"
    if label == "a":
        return "arithmetic"
    return "arithmetic"

def _conv_subtype(label: str) -> str:
    m = re.match(r"conv-([abcd])$", label)
    return m.group(1) if m else ""

def _op_from_problem(problem: str) -> str:
    if re.search(r"(?:\bx\b|\*|×)", problem):
        return "x"
    if "÷" in problem:
        return "÷"
    if "+" in problem:
        return "+"
    if "-" in problem:
        return "-"
    return ""

def _has_mixed_number(problem: str) -> bool:
    return bool(re.search(r"\b\d+\s+\d+/\d+\b", problem))

def _prec_from_op(op: str) -> str:
    if op in {"x"}:
        return "direct_multiplication"
    if op in {"+", "-"}:
        return "common_denominator_required_for_add_sub"
    return "none"

# ----------------------------
# Facet construction (label-aware, no label leakage)
# ----------------------------

_ALLOWED_OFAM = {"Conversion", "Property of Operation", "Property of Fractions", "Simplifying", "Arithmetic"}
_ALLOWED_OSTR = {"two_fractions", "fraction_and_integer", "mixed_number", "algebraic_fraction", "other"}
_ALLOWED_PREC = {"direct_multiplication", "common_denominator_required_for_add_sub", "none"}
_ALLOWED_TPIT = {
    "mixing numerator and denominator operations",
    "changing representations not called for",
    "combining numerators without first aligning denominators",
    "changing denominators inconsistently",
    "not applicable",
}

def _family_hints(label: str, problem: str):
    op = _op_from_problem(problem)
    preconditions = _prec_from_op(op)
    has_mixed = _has_mixed_number(problem)

    # Default
    ofam = "Arithmetic"
    ostr = "mixed_number" if has_mixed else "two_fractions"
    tpit = ["not applicable"]
    generation_note = "Treat this as general arithmetic without inferring correctness."

    fam = _family_from_label(label)

    if fam == "conversion":
        ofam = "Conversion"
        # subtype nuances
        sub = _conv_subtype(label)
        if sub == "a":
            tpit = ["changing representations not called for", "mixing numerator and denominator operations"]
            generation_note = "Representation change may combine parts before applying the written operation while preserving given values."
        elif sub == "b":
            tpit = ["mixing numerator and denominator operations", "changing denominators inconsistently"]
            generation_note = "Representation change may involve swap/inverse-like adjustments during setup while keeping given quantities unchanged."
        elif sub == "c":
            tpit = ["changing denominators inconsistently", "changing representations not called for"]
            generation_note = "Representation change may blend components additively when forming an equivalent form without altering written values."
        else:  # conv-d or other
            tpit = ["changing representations not called for"]
            generation_note = "Representation change should be reasoned about without altering given forms."

    elif fam == "property_of_operation":
        ofam = "Property of Operation"
        # keep neutral; operator-agnostic pitfalls
        tpit = ["not applicable"]

    elif fam == "property_of_fractions":
        ofam = "Property of Fractions"
        if op in {"+", "-"}:
            tpit = [
                "combining numerators without first aligning denominators",
                "changing denominators inconsistently",
            ]
        elif op == "x":
            tpit = ["mixing numerator and denominator operations"]
        else:
            tpit = ["not applicable"]

    elif fam == "arithmetic":
        ofam = "Arithmetic"
        tpit = ["not applicable"]

    # Clip to allowed vocab
    ofam = ofam if ofam in _ALLOWED_OFAM else "Arithmetic"
    ostr = ostr if ostr in _ALLOWED_OSTR else "other"
    preconditions = preconditions if preconditions in _ALLOWED_PREC else "none"
    tpit = [p for p in tpit if p in _ALLOWED_TPIT][:2] or ["not applicable"]

    return ofam, ostr, preconditions, tpit, generation_note

# ----------------------------
# Output sanitation / validation
# ----------------------------

def sanitize_context(s: str, original_problem: str) -> str:
    """
    Remove accidental 'final result' leaks like '= 1/3' or 'which is 1/3'.
    """
    s = re.sub(r"\b(which\s+is|which\s+equals|equals)\s+[^\.]+", "", s, flags=re.IGNORECASE)

    given_tokens = set(re.findall(r"\b\d+/\d+\b|\b\d+(?:\.\d+)?\b", original_problem))

    def _strip_new_results(match):
        token = match.group(1)
        return "" if token not in given_tokens else match.group(0)

    s = re.sub(r"=\s*([0-9]+/[0-9]+|[0-9]+(?:\.[0-9]+)?)", _strip_new_results, s)
    return re.sub(r"\s{2,}", " ", s).strip()

_REQUIRED_KEYS = {
    "problem", "student_answer", "operation_family", "operand_structure",
    "preconditions", "typical_pitfalls", "procedure_outline",
    "representation_notes", "context"
}

def _validate_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p)
    out.setdefault("problem", "The problem is provided as written.")
    out.setdefault("student_answer", "The student response is provided as written.")
    out.setdefault("operation_family", "Arithmetic")
    out.setdefault("operand_structure", "other")
    out.setdefault("preconditions", "none")
    if not isinstance(out.get("typical_pitfalls"), list) or not out["typical_pitfalls"]:
        out["typical_pitfalls"] = ["not applicable"]
    out.setdefault("procedure_outline", "Describe the high-level approach without executing any steps.")
    out.setdefault("representation_notes", "Preserve the given representations.")
    out.setdefault("context", "Provide a brief neutral description while preserving the original forms.")
    return out

# ----------------------------
# Prompts
# ----------------------------

CONTEXT_SYSTEM_PROMPT = (
    "You are a careful math tutor that produces brief, neutral context for a math item.\n"
    "HARD RULES:\n"
    "1) Do NOT compute or simplify; do NOT give or imply a final result.\n"
    "2) Preserve numbers/fractions exactly as written (e.g., '6/18' stays '6/18').\n"
    "3) Be neutral and descriptive; avoid judgment words or claims of correctness.\n"
    "4) No hedging about correctness (e.g., 'may indicate', 'suggests', etc.).\n"
    "5) Output must be valid JSON ONLY (no extra text) using the schema below.\n"
    "6) Each string field must be a single complete sentence ending with a period.\n"
    "7) Avoid repetition across fields.\n"
    "8) Keep 'context' to at most TWO sentences.\n"
    "9) Use only the allowed facet vocabulary below.\n"
    "\n"
    "ALLOWED FACETS:\n"
    "- operation_family: one of {Conversion, Property of Operation, Property of Fractions, Simplifying, Arithmetic}\n"
    "- operand_structure: one of {two_fractions, fraction_and_integer, mixed_number, algebraic_fraction, other}\n"
    "- preconditions: one of {direct_multiplication, common_denominator_required_for_add_sub, none}\n"
    "- typical_pitfalls: 1–2 items chosen from: "
    "{'mixing numerator and denominator operations', 'changing representations not called for', "
    "'combining numerators without first aligning denominators', 'changing denominators inconsistently', "
    "'not applicable'}\n"
    "\n"
    "SCHEMA (all fields required):\n"
    "{\n"
    '  "problem": "<restate problem exactly as one sentence.>",\n'
    '  "student_answer": "<restate student answer as one sentence.>",\n'
    '  "operation_family": "<from allowed list>",\n'
    '  "operand_structure": "<from allowed list>",\n'
    '  "preconditions": "<from allowed list>",\n'
    '  "typical_pitfalls": ["<from allowed list>", "<optional second from allowed list>"],\n'
    '  "procedure_outline": "<generic method as one sentence; no steps executed.>",\n'
    '  "representation_notes": "<preservation note as one sentence.>",\n'
    '  "context": "<two sentences max, neutral, no results.>"\n'
    "}\n"
)

def context_user_prompt(problem: str, student_answer: str, raw_row: str, label: str) -> str:
    # Build hints (fed to the model but not to be echoed as label names)
    ofam, ostr, prec, tpit, gnote = _family_hints(label, problem)
    hints_block = (
        "HINTS (do not reveal or name any label; do not judge correctness):\n"
        f"- operation_family_suggestion: {ofam}\n"
        f"- operand_structure_suggestion: {ostr}\n"
        f"- preconditions_suggestion: {prec}\n"
        f"- typical_pitfalls_suggestion: {', '.join(tpit)}\n"
        f"- generation_note: {gnote}\n"
    )

    few_shot = (
        "FEW-SHOT (STYLE ONLY; DO NOT COPY NUMBERS):\n"
        "---\n"
        "INPUT:\n"
        "Problem: 6/18 x 2/5\n"
        "Student Answer: 1/3\n"
        "OUTPUT:\n"
        "{\n"
        '  "problem": "The problem is 6/18 x 2/5.",\n'
        '  "student_answer": "The student response is 1/3.",\n'
        '  "operation_family": "Property of Fractions",\n'
        '  "operand_structure": "two_fractions",\n'
        '  "preconditions": "direct_multiplication",\n'
        '  "typical_pitfalls": ["mixing numerator and denominator operations"],\n'
        '  "procedure_outline": "Consider multiplication of two fractions by attending to numerators and denominators as written.",\n'
        '  "representation_notes": "Keep 6/18 and 2/5 exactly as given without altering their forms.",\n'
        '  "context": "This item involves multiplying two fractions with direct multiplication. Focus on how numerators and denominators are combined as written while preserving the original forms."\n'
        "}\n"
        "---\n"
    )

    return (
        "Produce JSON that follows the schema and hard rules exactly. "
        "Fill all fields. Do not compute or simplify. Use only allowed facet vocabulary.\n\n"
        f"{hints_block}\n"
        f"Raw row: {raw_row}\n"
        f"Problem: {problem if problem else '(unknown)'}\n"
        f"Student Answer: {student_answer if student_answer else '(unknown)'}\n\n"
        f"{few_shot}"
        "RETURN ONLY THE JSON FOR THE CURRENT INPUT."
    )

# ----------------------------
# OpenAI wrapper
# ----------------------------

def complete_with_gpt(model: str, system_prompt: str, user_prompt: str) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

# ----------------------------
# Row processor
# ----------------------------

def process_row(text: str, label: str, model: str, dry_run: bool=False
               ) -> Tuple[str, str, str, str, str, str, str]:
    prob_raw, ans_raw = extract_problem_answer(text)
    prob_raw = prob_raw or ""
    ans_raw  = ans_raw or ""

    problem_latex = text_to_latex(prob_raw)
    answer_latex  = text_to_latex(ans_raw)
    problem_opt   = text_to_opt(prob_raw)
    answer_opt    = text_to_opt(ans_raw)

    if dry_run:
        # Build a deterministic, non-computational payload from hints
        ofam, ostr, prec, tpit, _ = _family_hints(label, prob_raw)
        payload = {
            "problem": f"The problem is {prob_raw}.",
            "student_answer": f"The student response is {ans_raw}.",
            "operation_family": ofam,
            "operand_structure": ostr,
            "preconditions": prec,
            "typical_pitfalls": tpit,
            "procedure_outline": "Describe the high-level approach without executing any steps.",
            "representation_notes": "Preserve all given numbers and fraction forms.",
            "context": "Provide a brief neutral description while preserving the original forms."
        }
    else:
        try:
            resp = complete_with_gpt(
                model,
                CONTEXT_SYSTEM_PROMPT,
                context_user_prompt(prob_raw, ans_raw, text, label)
            )
            payload = json.loads(resp)
        except Exception as e:
            # Fallback safe payload
            ofam, ostr, prec, tpit, _ = _family_hints(label, prob_raw)
            payload = {
                "problem": f"The problem is {prob_raw}.",
                "student_answer": f"The student response is {ans_raw}.",
                "operation_family": ofam,
                "operand_structure": ostr,
                "preconditions": prec,
                "typical_pitfalls": tpit,
                "procedure_outline": "Describe the high-level approach without executing any steps.",
                "representation_notes": "Preserve all given numbers and fraction forms.",
                "context": "Provide a brief neutral description while preserving the original forms."
            }

    payload = _validate_payload(payload)

    # Compact facet line (never mention the label)
    ofam = payload["operation_family"]
    ostr = payload["operand_structure"]
    prec = payload["preconditions"]
    tpit_list = [p for p in payload.get("typical_pitfalls", []) if p and p != "not applicable"][:2]
    tpit_str = "; ".join(tpit_list) if tpit_list else "not applicable"

    # Two-sentence context: (1) restate problem, (2) restate student answer
    ctx_two_sentences = f"{payload['problem']} {payload['student_answer']}".strip()
    facet_line = f"[OFAM] {ofam} [OSTR] {ostr} [PREC] {prec} [TPIT] {tpit_str}".strip()

    context_gpt = sanitize_context(f"{ctx_two_sentences}\n{facet_line}", prob_raw)

    # Final assembly for Math-BERT ingestion
    a_segment = f"[FORM] {problem_latex} [ANS] {answer_latex}".strip()
    b_parts = [f"[CTX] {context_gpt}"]
    if problem_opt:
        b_parts.append(f"[OPT] {problem_opt}")
    if answer_opt:
        b_parts.append(f"[OPT_ANS] {answer_opt}")
    b_segment = " ".join(b_parts).strip()

    final_text = f"{a_segment} [SEP] {b_segment}".strip()

    return (prob_raw, ans_raw,
            problem_latex, answer_latex,
            context_gpt,
            problem_opt, answer_opt)

# ----------------------------
# Example batch driver
# ----------------------------

if __name__ == "__main__":
    # Example small DataFrame substitute
    # If you already have df, just replace this block with your df loop.
    try:
        import pandas as pd
        df = pd.DataFrame([
            {"input": "Problem: 6/18 x 2/5 | Student Answer: 1/3", "label": "conv-a"},
            {"input": "Problem: 6/18 x 2/5 | Student Answer: 5/6", "label": "conv-b"},
            {"input": "Problem: 6/18 x 2/5 | Student Answer: 8/15", "label": "conv-c"},
            {"input": "Problem: 4/18 - 1/7 | Student Answer: 23/25", "label": "pf-a"},
        ])
    except Exception:
        df = None

    rows = []
    model = "gpt-4o-mini"
    dry_run = not _HAS_OPENAI  # auto-fallback to dry_run if no key
    sleep_time = 0.3

    if df is not None:
        for i, row in df.iterrows():
            text = str(row["input"])
            label = str(row["label"])

            (prob_raw, ans_raw,
             problem_latex, answer_latex,
             context_gpt,
             problem_opt, answer_opt) = process_row(text, label, model, dry_run=dry_run)

            final_text = (
                f"[FORM] {problem_latex} [ANS] {answer_latex} "
                f"[SEP] [CTX] {context_gpt}"
                f"{' [OPT] ' + problem_opt if problem_opt else ''}"
                f"{' [OPT_ANS] ' + answer_opt if answer_opt else ''}"
            ).strip()

            rows.append({
                "input": text,
                "label": label,
                "prob_raw": prob_raw,
                "ans_raw": ans_raw,
                "problem_latex": problem_latex,
                "answer_latex": answer_latex,
                "problem_opt": problem_opt,
                "answer_opt": answer_opt,
                "context_gpt": context_gpt,
                "final_text": final_text
            })

            print(i, prob_raw, "||", ans_raw)
            if sleep_time:
                time.sleep(sleep_time)

        try:
            import pandas as pd
            out_df = pd.DataFrame(rows)
            print("\nSAMPLE OUTPUT ROWS:\n", out_df.head(3).to_dict(orient="records"))
        except Exception:
            print("\nSAMPLE OUTPUT ROWS (first 3):\n", rows[:3])