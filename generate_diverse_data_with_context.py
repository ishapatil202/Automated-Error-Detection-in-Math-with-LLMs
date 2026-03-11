# mathbert_diverse_generator_with_error_context.py
# Compatible with tbs17/MathBERT; includes error-aware context
import random
import re
import pandas as pd
from fractions import Fraction

# --- Helper Functions ---
def random_fraction():
    den = random.randint(2, 20)
    num = random.randint(1, den-1)
    return num, den

def random_mixed():
    whole = random.randint(1, 5)
    num, den = random_fraction()
    return whole, num, den

def to_improper(whole, num, den):
    return whole * den + num, den

def format_fraction(num, den):
    if den == 0:
        return "Undefined"
    f = Fraction(num, den)
    num, den = f.numerator, f.denominator
    if abs(num) >= den and den != 1:
        whole = abs(num) // den
        rem = abs(num) % den
        sign = '-' if num < 0 else ''
        if rem == 0:
            return f"{sign}{whole}"
        else:
            return f"{sign}{whole} {rem}/{den}"
    else:
        return f"{num}/{den}"

def random_operand():
    typ = random.choice(['mixed', 'proper', 'improper'])
    if typ == 'mixed':
        w, n, d = random_mixed()
        return typ, (w, n, d)
    elif typ == 'proper':
        n, d = random_fraction()
        return typ, (n, d)
    else:
        d = random.randint(2, 20)
        n = random.randint(d, 2*d)
        return typ, (n, d)

def operand_to_improper(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return w * d + n, d
    else:
        n, d = value
        return n, d

def operand_to_string(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return f"{w} {n}/{d}"
    else:
        n, d = value
        return f"{n}/{d}"

# ============================
# LaTeX + OPT builders (MathBERT-friendly)
# ============================

# LaTeX operator map
_OP_TO_LATEX = {"+": "+", "-": "-", "x": r"\times", "÷": r"\div"}

def latex_of_operand(typ, value) -> str:
    """Compact LaTeX for an operand, preserving mixed/proper/improper form."""
    if typ == "mixed":
        w, n, d = value
        return rf"{w}\,\frac{{{n}}}{{{d}}}"
    else:
        n, d = value
        return rf"\frac{{{n}}}{{{d}}}"

def latex_of_problem(op1_type, op1_value, op, op2_type, op2_value) -> str:
    left  = latex_of_operand(op1_type, op1_value)
    right = latex_of_operand(op2_type, op2_value)
    oplx  = _OP_TO_LATEX[op]
    return rf"{left} \, {oplx} \, {right}"

def latex_of_answer_from_str(ans: str) -> str:
    """Render your formatted answer string to LaTeX (int, mixed, fraction, Undefined)."""
    s = ans.strip()
    if s.lower() == "undefined":
        return r"\text{Undefined}"
    # integer
    m = re.match(r"^[+-]?\d+$", s)
    if m:
        return s
    # mixed: optional sign + whole + space + num/den
    m = re.match(r"^([+-]?)(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        sign, whole, num, den = m.groups()
        sign = "-" if sign == "-" else ""
        return rf"{sign}{int(whole)}\,\frac{{{int(num)}}}{{{int(den)}}}"
    # fraction: optional signed numerator
    m = re.match(r"^([+-]?\d+)/(\d+)$", s)
    if m:
        num, den = m.groups()
        if num.startswith("-"):
            return rf"-\frac{{{abs(int(num))}}}{{{int(den)}}}"
        return rf"\frac{{{int(num)}}}{{{int(den)}}}"
    return rf"\text{{{s}}}"

# OPT operator map (prefix binary ops); mixed numbers decomposed for compatibility
_OP_TO_OPT = {"+": "ADD", "-": "SUB", "x": "MUL", "÷": "DIV"}

def opt_of_operand_expr(typ, value) -> str:
    """
    Build OPT for a single operand.
    Mixed number w n/d -> ADD INT w FRAC n d  (no custom MIX token).
    """
    if typ == "mixed":
        w, n, d = value
        return f"ADD INT {w} FRAC {n} {d}"
    else:
        n, d = value
        return f"FRAC {n} {d}"

def opt_of_problem(op1_type, op1_value, op, op2_type, op2_value) -> str:
    """Prefix operator tree for the LHS problem only."""
    op_tok = _OP_TO_OPT[op]
    left   = opt_of_operand_expr(op1_type, op1_value)
    right  = opt_of_operand_expr(op2_type, op2_value)
    return f"{op_tok} {left} {right}"

# ---------- Student answer -> OPT (NO MIX) ----------
# MIX w n/d  -->  ADD INT w FRAC n d
# - (mixed)  -->  SUB INT 0 (ADD INT w FRAC n d)
def student_answer_to_opt_nomix(ans: str) -> str:
    s = ans.strip()
    if s.lower() == "undefined":
        return "UNDEF"

    # integer
    m = re.match(r"^[+-]?\d+$", s)
    if m:
        return f"INT {int(s)}"

    # mixed: optional sign + whole + space + num/den
    m = re.match(r"^([+-]?)(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        sign, whole, num, den = m.groups()
        base = f"ADD INT {int(whole)} FRAC {int(num)} {int(den)}"
        if sign == "-":
            return f"SUB INT 0 {base}"
        else:
            return base

    # fraction: optional signed numerator
    m = re.match(r"^([+-]?\d+)/(\d+)$", s)
    if m:
        num, den = m.groups()
        return f"FRAC {int(num)} {int(den)}"

    # fallback (shouldn't happen with your format_fraction)
    return f"ANS {s}"

# ---------- Build equation OPT with EQ (infix), NO MIX ----------
def build_prob_ans_opt_nomix(problem_opt: str, student_ans: str):
    answer_opt = student_answer_to_opt_nomix(student_ans)
    prob_ans_opt = f"{problem_opt} EQ {answer_opt}"
    return prob_ans_opt, answer_opt

# ---------- Safe input_text builder (problem+answer LaTeX + context only) ----------
def build_input_text_mathbert(formula_latex: str, context_text: str) -> str:
    # Do NOT append OPT to the model input (MathBERT expects LaTeX+context)
    return f"[CLS] {formula_latex} [SEP] {context_text} [SEP]"

# ---------- Error family/type -> diagnostic context sentence ----------
# Derived from the rubric in your screenshots (Conversion, Property of Operation, Property of Fractions, Arithmetic)
def error_diagnostic_sentence(label: str, op_symbol: str) -> str:
    # op_symbol is one of '+', '-', 'x', '÷' (helps pick wording)
    op_word = {'+': 'addition', '-': 'subtraction', 'x': 'multiplication', '÷': 'division'}[op_symbol]
    diag = {
        # Conversion (replacing whole/mixed with fractions or other forms inappropriately)
        "conv-a": f"Likely a conversion issue: bundling whole and fractional parts, then operating across them during {op_word}.",
        "conv-b": f"Likely a conversion issue: using a reciprocal/cross-inversion step during setup rather than performing {op_word} correctly.",
        "conv-c": f"Likely a conversion issue: altering representations (e.g., combining parts) before aligning terms required for {op_word}.",
        "conv-d": f"Conversion behavior present but not matching common patterns; representation changed before carrying out {op_word}.",

        # Property of Operation (misapplied additive/multiplicative inverse, distributive property, etc.)
        "po-a":  "Misapplied operation property: an additive-inverse transformation seems to have been introduced.",
        "po-b":  "Misapplied operation property: a multiplicative inverse/reciprocal was applied to transform the expression.",
        "po-c":  f"Misapplied operation property: a distributive-like rewrite over fractional parts appears to have been used for {op_word}.",
        "po-d":  "Other property-of-operation handling that does not follow the standard algorithm.",

        # Property of Fractions (adjustment/renaming errors: denominators/numerators)
        "pf-a":  f"Denominator issue: operated across denominators or failed to establish a proper common denominator before {op_word}.",
        "pf-b":  f"Denominator issue: incorrectly chose one denominator as if common (or renamed fractions inconsistently) for {op_word}.",
        "pf-c":  "Numerator issue: operated on numerators in isolation from denominators.",
        "pf-d":  "Other numerator-only handling that breaks the fraction-computation procedure.",
        "pf-e":  f"Used denominator product in a context of {op_word} where equal denominators are required.",

        # Arithmetic (local integer/slip errors)
        "a":     "Arithmetic slip: local integer/computation error likely occurred."
    }
    # Fallback if an unknown key arrives
    return diag.get(label, "Computation appears to follow a non-standard pattern for this operation.")

# --- Error Generation Functions (same as before) ---
def conv_a_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d1 + d2)

def conv_b_error(n1, d1, n2, d2):
    choice = random.choice([
        lambda: format_fraction(n1 * n2, d1 * d2),
        lambda: format_fraction(n1 * d2, d1 * n2),
        lambda: format_fraction(n2, d1),
        lambda: format_fraction(n1, d2),
    ])
    return choice()

def conv_c_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d1)

def conv_d_error(n1, d1, n2, d2):
    return format_fraction(n1 - n2, d1 + d2)

def po_a_error(correct_num, correct_den):
    return format_fraction(-correct_num, correct_den)

def po_b_error(correct_num, correct_den):
    if correct_num != 0:
        return format_fraction(correct_den, correct_num)
    else:
        return "Undefined"

def po_c_error(n1, d1, n2, d2):
    return format_fraction(n1 * n2, d1 + d2)

def po_d_error(n1, d1, n2, d2):
    den = d1 - d2 if n1 or n2 else 1  # defensive
    den = d1 - d2 if d1 != d2 else 1
    return format_fraction(n1 + n2, den)

def pf_a_error(n1, d1, n2, d2):
    return format_fraction(n1, d1 + d2)

def pf_b_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d2)

def pf_c_error(n1, d1, n2, d2):
    return format_fraction(n1 * n2, d1)

def pf_d_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d2)

def pf_e_error(n1, d1, n2, d2):
    return format_fraction(n1 - n2, d1 * d2)

def a_error(num, den):
    return format_fraction(num + 1, den)

# --- Main Data Generation ---
data = []
labels = [
    ("conv-a", conv_a_error),
    ("conv-b", conv_b_error),
    ("conv-c", conv_c_error),
    ("conv-d", conv_d_error),
    ("po-a", po_a_error),
    ("po-b", po_b_error),
    ("po-c", po_c_error),
    ("po-d", po_d_error),
    ("pf-a", pf_a_error),
    ("pf-b", pf_b_error),
    ("pf-c", pf_c_error),
    ("pf-d", pf_d_error),
    ("pf-e", pf_e_error),
    ("a", a_error),
]

unique_problems = set()
target_count = 30000

while len(data) < target_count:
    op1_type, op1_value = random_operand()
    op2_type, op2_value = random_operand()
    op = random.choice(['+', '-', 'x', '÷'])

    op1_str = operand_to_string(op1_type, op1_value)
    op2_str = operand_to_string(op2_type, op2_value)
    problem = f"{op1_str} {op} {op2_str}"

    n1, d1 = operand_to_improper(op1_type, op1_value)
    n2, d2 = operand_to_improper(op2_type, op2_value)

    if op == '+':
        num = n1 * d2 + n2 * d1
        den = d1 * d2
    elif op == '-':
        num = n1 * d2 - n2 * d1
        den = d1 * d2
    elif op == 'x':
        num = n1 * n2
        den = d1 * d2
    else:
        num = n1 * d2
        den = d1 * n2 if n2 != 0 else 1

    key = (problem, n1, d1, n2, d2, op)
    if key in unique_problems:
        continue
    unique_problems.add(key)

    correct_frac = Fraction(num, den)
    correct_num = correct_frac.numerator
    correct_den = correct_frac.denominator

    # Build once per (problem, op) for shared columns
    problem_latex = latex_of_problem(op1_type, op1_value, op, op2_type, op2_value)
    problem_opt   = opt_of_problem(op1_type, op1_value, op, op2_type, op2_value)

    for label, err_func in labels:
        if label in ["po-a", "po-b", "a"]:
            student_ans = err_func(correct_num, correct_den)
        else:
            student_ans = err_func(n1, d1, n2, d2)

        # --- LaTeX segments (shown to MathBERT) ---
        answer_latex  = latex_of_answer_from_str(student_ans)
        formula_latex = rf"{problem_latex} \,=\, {answer_latex}"

        # --- Error-aware context (three sentences) ---
        # 1) problem statement
        # 2) student response
        # 3) brief diagnostic hint derived from label & screenshots (no raw tag leakage)
        diag_sentence = error_diagnostic_sentence(label, op)
        context_text  = (
            f"The problem is {problem}. "
            f"The student response is {student_ans}. "
            f"{diag_sentence}"
        )

        # --- OPT (analysis/supervision only; NOT fed into model input) ---
        prob_ans_opt, answer_opt = build_prob_ans_opt_nomix(problem_opt, student_ans)

        # --- Build model input (safe) ---
        input_text = build_input_text_mathbert(formula_latex, context_text)

        # --- Row: preserve existing columns, add corrected ones ---
        row = {
            "input": input_text,   # LaTeX (problem=answer) + error-aware context
            "label": label,

            # helpful extras for auditing / later tasks
            "formula_latex": formula_latex,
            "context_text":  context_text,
            "problem_opt":   problem_opt,
            "answer_latex":  answer_latex,
            "prob_ans_latex": rf"{formula_latex}",
            "answer_opt":    answer_opt,
            "prob_ans_opt":  prob_ans_opt,
            "problem_raw":   problem,
        }

        # Guardrail: ensure no MIX in OPTs
        if (" MIX " in f" {answer_opt} ") or (" MIX " in f" {prob_ans_opt} ") or ("-MIX" in f"{answer_opt}{prob_ans_opt}"):
            raise ValueError(f"MIX detected in OPT: answer_opt={answer_opt}, prob_ans_opt={prob_ans_opt}")

        data.append(row)
        if len(data) >= target_count:
            break

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("diversed-data.csv", index=False)
print(f"Generated {len(df)} rows in 'diversed-data.csv' (MathBERT-compatible input with error-aware context).")