#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathBERT ingest pipeline:
- Read rows from a CSV with columns: input, label
- For each row:
    * Extract math expression(s) heuristically
    * Convert to SymPy expression(s) and LaTeX
    * Traverse the SymPy tree(s) to produce OPT-like operator tokens
    * Prompt a GPT model to write a short, pedagogical context paragraph
- Write a new CSV with columns:
    input, label, form_latex, context_gpt, opt_tokens, final_text

Usage:
    pip install pandas sympy tenacity openai
    export OPENAI_API_KEY=sk-...
    python mathbert_ingest_pipeline.py \
        --in_csv improved_diversed_data.csv \
        --out_csv mathbert_ready.csv \
        --model gpt-5.1 \
        --max_rows 50000 \
        --dry_run false

Notes:
- This script assumes a GPT-5 class model is available to your account via the OpenAI SDK.
- If you use a different provider or SDK, replace `complete_with_gpt()` accordingly.
"""

import os
import re
import csv
import json
import time
import argparse
from typing import List, Tuple, Optional

import pandas as pd
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from sympy import sympify, Eq
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.printing.latex import latex
from sympy import Symbol, Function
from sympy.core import Add, Mul, Pow, Rational, Integer, Float, Symbol
from sympy import Eq as SymEq

# ---------- OpenAI client (replace if needed) ----------
try:
    # Newer OpenAI SDK style:
    from openai import OpenAI
    _client = OpenAI()
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    _client = None


# ---------- Heuristics to extract an expression from a row ----------

EXPR_PATTERNS = [
    r"Problem:\s*(.*?)\s*\|\s*Student Answer:\s*(.*)",   # Problem/Answer pattern
    r"Expr:\s*(.*)",                                     # Expr: ... pattern
]

def extract_problem_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Try explicit "Problem: ... | Student Answer: ..."
    m = re.search(EXPR_PATTERNS[0], text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None

def extract_any_expr(text: str) -> Optional[str]:
    # Fall back: pick the longest inline math-like span (digits/operators/variables)
    candidates = re.findall(r"[A-Za-z0-9\s\^\*/\+\-\(\)=\.\,\\frac\\times\\cdot]+", text)
    candidates = [c.strip() for c in candidates if len(c.strip()) >= 3]
    if not candidates:
        return None
    return max(candidates, key=len)


# ---------- SymPy parsing and LaTeX ----------

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def to_sympy(expr_str: str):
    """Convert a string like '6/18 x 2/5 = 1/3' into a SymPy object.
    We'll normalize common variants (e.g., 'x' -> '*') and allow equals."""
    if expr_str is None:
        return None
    s = expr_str
    # Normalize common math tokens
    s = s.replace("×", "*").replace("x", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("^", "**")
    # Allow 'a=b' as equation
    if "=" in s:
        parts = s.split("=")
        if len(parts) == 2:
            left = parse_expr(parts[0], transformations=_TRANSFORMS)
            right = parse_expr(parts[1], transformations=_TRANSFORMS)
            return SymEq(left, right)
        # if multiple equals, fall back to parse the left-most chunk
        s = parts[0]
    # Plain expression
    try:
        return parse_expr(s, transformations=_TRANSFORMS)
    except Exception:
        return None

def to_latex(expr) -> Optional[str]:
    try:
        return latex(expr)
    except Exception:
        return None


# ---------- Build OPT-like operator tokens by traversing SymPy ----------

def sympy_to_opt(expr) -> List[str]:
    """Preorder traversal producing tokens like ADD, MUL, FRAC, SUP, EQ, CONST, VAR, ..."""
    tokens: List[str] = []
    def visit(e):
        # Numbers
        if isinstance(e, (Integer, Float)):
            tokens.append(str(e))
            return
        if isinstance(e, Rational):
            tokens.extend(["FRAC", str(e.p), str(e.q)])
            return
        # Variables / symbols
        if isinstance(e, Symbol):
            tokens.append(str(e))
            return
        # Operations
        if isinstance(e, Add):
            tokens.append("ADD")
            for arg in e.args:
                visit(arg)
            return
        if isinstance(e, Mul):
            tokens.append("MUL")
            for arg in e.args:
                visit(arg)
            return
        if isinstance(e, Pow):
            tokens.append("SUP")  # exponent/superscript
            base, exp = e.args
            visit(base); visit(exp)
            return
        if isinstance(e, SymEq):
            tokens.append("EQ")
            visit(e.lhs); visit(e.rhs)
            return
        # Fallback: print class name then children
        tokens.append(type(e).__name__.upper())
        if hasattr(e, "args"):
            for arg in e.args:
                visit(arg)
    visit(expr)
    return tokens


# ---------- GPT context generation ----------

CONTEXT_SYSTEM_PROMPT = (
    "You are a helpful math tutor. Given a math problem and an optional student answer, "
    "write a 1-3 sentence context that explains the key idea to solve/check it. "
    "Be concise, neutral, and do not reveal the final answer unless it is already present."
)

def context_user_prompt(problem: Optional[str], student_answer: Optional[str], raw_row: str) -> str:
    return (
        "Generate a short context for the following.\n"
        f"Raw row: {raw_row}\n"
        f"Problem: {problem if problem else '(unknown)'}\n"
        f"Student Answer: {student_answer if student_answer else '(unknown)'}\n"
        "Reply in strict JSON with a single key 'context'. Example: {\"context\": \"...\"}"
    )

@retry(wait=wait_exponential_jitter(initial=1, max=30), stop=stop_after_attempt(6))
def complete_with_gpt(model: str, system_prompt: str, user_prompt: str) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY.")
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


# ---------- Main pipeline ----------

def process_row(text: str, model: str, dry_run: bool=False) -> Tuple[str, str, str]:
    """Return (form_latex, context_gpt, opt_tokens_str)"""

    # 1) Try to split into Problem / Student Answer
    prob, ans = extract_problem_and_answer(text)

    # 2) Extract a parseable expression (priority: problem, else whole text)
    expr_candidate = prob if prob else extract_any_expr(text)
    expr = to_sympy(expr_candidate) if expr_candidate else None

    # 3) Build LaTeX
    form_latex = to_latex(expr) if expr is not None else ""

    # 4) Build OPT tokens
    opt_tokens = " ".join(sympy_to_opt(expr)) if expr is not None else ""

    # 5) Ask GPT for short context
    if dry_run:
        context_gpt = "This problem involves operating on fractions and checking equivalence; reduce to lowest terms to verify."
    else:
        resp = complete_with_gpt(model, CONTEXT_SYSTEM_PROMPT, context_user_prompt(prob, ans, text))
        try:
            context_gpt = json.loads(resp)["context"]
        except Exception:
            # fallback: keep raw response
            context_gpt = resp

    return form_latex, context_gpt, opt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to input CSV with columns: input, label")
    ap.add_argument("--out_csv", required=True, help="Path to output CSV to write")
    ap.add_argument("--model", default="gpt-5.1", help="GPT model name")
    ap.add_argument("--max_rows", type=int, default=None, help="Limit rows for a quick run")
    ap.add_argument("--dry_run", type=lambda s: s.lower()=="true", default="false", help="If true, skip API calls and use a canned context")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between API calls (seconds)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert {"input","label"} <= set(df.columns), "CSV must have columns: input,label"

    rows = []
    n = len(df) if args.max_rows is None else min(len(df), args.max_rows)
    for i in range(n):
        row = df.iloc[i]
        text = str(row["input"])
        label = row["label"]

        try:
            form_latex, context_gpt, opt_tokens = process_row(text, args.model, dry_run=args.dry_run)
        except Exception as e:
            # On error, record blanks so you can filter/retry later
            form_latex, context_gpt, opt_tokens = "", f"[ERROR] {e}", ""

        final_text = f"[FORM] {form_latex} [SEP] {context_gpt} [SEP] {opt_tokens}".strip()

        rows.append({
            "input": text,
            "label": label,
            "form_latex": form_latex,
            "context_gpt": context_gpt,
            "opt_tokens": opt_tokens,
            "final_text": final_text
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

        if (i+1) % 200 == 0:
            print(f"Processed {i+1} / {n} rows")

    out_df = pd.DataFrame(rows, columns=["input","label","form_latex","context_gpt","opt_tokens","final_text"])
    out_df.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
