import random
import pandas as pd
from fractions import Fraction
import numpy as np

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

def operand_to_string(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return f"{w} {n}/{d}"
    else:
        n, d = value
        return f"{n}/{d}"

def operand_to_improper(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return w * d + n, d
    else:
        n, d = value
        return n, d

# --- CORRECTED Error Generation Functions (Following Official Definitions) ---
def generate_conv_a_errors(n1, d1, n2, d2, op):
    """Official conv-a: a b/c = X/Y, where X = (a+c)×b, a+b+c, a+b, a×b, or a/c"""
    # For mixed number conversion errors
    patterns = [
        lambda: format_fraction((n1 + d1) * n2, d1 * d2),  # (a+c)×b
        lambda: format_fraction(n1 + n2 + d1, d1 + d2),    # a+b+c
        lambda: format_fraction(n1 + n2, d1 + d2),         # a+b
        lambda: format_fraction(n1 * n2, d1 * d2),         # a×b
        lambda: format_fraction(n1, d1)                    # a/c
    ]
    return random.choice(patterns)()

def generate_conv_b_errors(n1, d1, n2, d2, op):
    """Official conv-b: Multiplicative inverse during conversion - a×b/a×c, a×b/b×c, b/(a×b)+c"""
    if op == '÷':
        # Common mistake: multiply instead of divide during conversion
        return format_fraction(n1 * n2, d1 * d2)
    elif op == 'x':
        # Common mistake: divide instead of multiply during conversion
        # FIXED: Use integer division to avoid float issues
        if n2 != 0 and d2 != 0:
            return format_fraction(n1 * d2, n2 * d1)  # Cross multiply to avoid floats
        else:
            return "Undefined"
    else:
        # For addition/subtraction: use multiplicative patterns during conversion
        patterns = [
            lambda: format_fraction(n1 * n2, n1 * d2),  # a×b/a×c
            lambda: format_fraction(n1 * n2, n2 * d1),  # a×b/b×c
            lambda: format_fraction(n2 + d1 * (n1 * n2), n1 * n2)  # FIXED: b/(a×b)+c
        ]
        return random.choice(patterns)()

def generate_conv_c_errors(n1, d1, n2, d2, op):
    """Official conv-c: a b/c = X, where X = a×c+b, (a+c)×b, a+b+c"""
    patterns = [
        lambda: format_fraction(n1 * d1 + n2, d1 * d2),  # a×c+b
        lambda: format_fraction((n1 + d1) * n2, d1 * d2), # (a+c)×b
        lambda: format_fraction(n1 + n2 + d1, d1 + d2)    # a+b+c
    ]
    return random.choice(patterns)()

def generate_conv_d_errors(n1, d1, n2, d2, op):
    """Official conv-d: Other conversion errors"""
    if op == '-':
        # Mistake: subtract numerators and denominators
        return format_fraction(n1 - n2, d1 - d2) if d1 != d2 else format_fraction(n1 - n2, 1)
    else:
        # Other conversion mistakes
        return format_fraction(n1 - n2, d1 + d2)

def generate_po_a_errors(correct_num, correct_den):
    """Official po-a: Additive inverse property (e.g., -a/b = a/b, a - d = d - a)"""
    return format_fraction(-correct_num, correct_den)

def generate_po_b_errors(correct_num, correct_den):
    """Official po-b: Multiplicative inverse property (e.g., 1÷n = n, X÷b/c = X×b/c)"""
    if correct_num != 0:
        return format_fraction(correct_den, correct_num)
    else:
        return "Undefined"

def generate_po_c_errors(n1, d1, n2, d2, op):
    """Official po-c: Distributive property (e.g., -a/b = a + b/c)"""
    if op == '÷':
        # Do addition instead of division
        return format_fraction(n1 + n2, d1 + d2)
    elif op == 'x':
        # Do subtraction instead of multiplication
        return format_fraction(n1 - n2, d1 - d2) if d1 != d2 else format_fraction(n1 - n2, 1)
    else:
        # For addition/subtraction: do multiplication
        return format_fraction(n1 * n2, d1 * d2)

def generate_po_d_errors(n1, d1, n2, d2, op):
    """Official po-d: Other property of operation errors"""
    if op == '÷':
        # Do multiplication with inverse
        return format_fraction(n1 * d2, d1 * n2) if n2 != 0 else "Undefined"
    else:
        # Other property errors
        return format_fraction(n1 + n2, d1 * d2)

def generate_pf_a_errors(n1, d1, n2, d2, op):
    """Official pf-a: Denominator issue 1 - incorrectly operating denominators (add/subtract/divide across denominators)"""
    if op in ['+', '-']:
        patterns = [
            lambda: format_fraction(n1 + n2, d1 + d2),  # add denominators
            lambda: format_fraction(n1 + n2, d1 - d2) if d1 != d2 else format_fraction(n1 + n2, 1),  # subtract denominators
            lambda: format_fraction((n1 + n2) * d2, d1) if d2 != 0 else "Undefined"  # FIXED: divide denominators
        ]
        return random.choice(patterns)()
    else:
        # For multiplication/division: add/subtract/divide across denominators
        return format_fraction(n1 * n2, d1 + d2)

def generate_pf_b_errors(n1, d1, n2, d2, op):
    """Official pf-b: Denominator issue 2 - incorrectly choosing denominator"""
    if op in ['+', '-']:
        # Choose one denominator (unless one is multiple of other)
        if d1 % d2 == 0 or d2 % d1 == 0:
            chosen_den = max(d1, d2)  # choose larger if one is multiple
        else:
            chosen_den = random.choice([d1, d2])  # choose one randomly
        return format_fraction(n1 + n2, chosen_den)
    else:
        # For multiplication/division: choose equal denominator
        return format_fraction(n1 * n2, d1)  # use first denominator

def generate_pf_c_errors(n1, d1, n2, d2, op):
    """Official pf-c: Numerator issue - inappropriately operated numerators"""
    if op in ['+', '-']:
        patterns = [
            lambda: format_fraction(n1 * n2, d1 + d2),  # multiply numerators
            lambda: format_fraction(n1 * d2, n2 * (d1 + d2)) if n2 != 0 else "Undefined",  # FIXED: divide numerators
            lambda: format_fraction(n1 - n2, d1 + d2)  # subtract numerators
        ]
        return random.choice(patterns)()
    else:
        # For multiplication/division
        return format_fraction(n1 + n2, d1 * d2)

def generate_pf_d_errors(n1, d1, n2, d2, op):
    """Official pf-d: Numerator issue - inappropriately operated numerators"""
    if op in ['+', '-']:
        # Different numerator operation patterns
        patterns = [
            lambda: format_fraction(n1 + n2, d2),  # use second denominator
            lambda: format_fraction(n1 * n2, d1),  # multiply numerators, use first denominator
            lambda: format_fraction(n1 - n2, d1)   # subtract numerators, use first denominator
        ]
        return random.choice(patterns)()
    else:
        # For multiplication/division
        return format_fraction(n1, d1 + d2)

def generate_pf_e_errors(n1, d1, n2, d2, op):
    """Official pf-e: Other fraction property errors"""
    if op == '-':
        # Multiply denominators in subtraction
        return format_fraction(n1 - n2, d1 * d2)
    else:
        # Other fraction property mistakes
        return format_fraction(n1 + n2, d1 - d2) if d1 != d2 else format_fraction(n1 + n2, 1)


def generate_a_errors(num, den):
    """Official a: Arithmetic errors - integer calculation"""
    # Simple arithmetic mistakes
    patterns = [
        lambda: format_fraction(num + 1, den),      # add 1 to numerator
        lambda: format_fraction(num, den + 1),      # add 1 to denominator
        lambda: format_fraction(num - 1, den),      # subtract 1 from numerator
        lambda: format_fraction(num * 2, den)       # multiply numerator by 2
    ]
    return random.choice(patterns)()

# --- Main Data Generation ---
def generate_improved_dataset(target_count=50000):
    data = []
    unique_problems = set()
    
    # Error generation functions with their specific patterns
    error_functions = {
        "conv-a": generate_conv_a_errors,
        "conv-b": generate_conv_b_errors,
        "conv-c": generate_conv_c_errors,
        "conv-d": generate_conv_d_errors,
        "po-a": generate_po_a_errors,
        "po-b": generate_po_b_errors,
        "po-c": generate_po_c_errors,
        "po-d": generate_po_d_errors,
        "pf-a": generate_pf_a_errors,
        "pf-b": generate_pf_b_errors,
        "pf-c": generate_pf_c_errors,
        "pf-d": generate_pf_d_errors,
        "pf-e": generate_pf_e_errors,
        "a": generate_a_errors,
    }
    
    while len(data) < target_count:
        # Generate diverse problem types
        op1_type = random.choice(['mixed', 'proper', 'improper'])
        op2_type = random.choice(['mixed', 'proper', 'improper'])
        operation = random.choice(['+', '-', 'x', '÷'])
        
        # Generate operands
        if op1_type == 'mixed':
            w1, n1, d1 = random_mixed()
            op1_str = f"{w1} {n1}/{d1}"
            n1_imp, d1_imp = to_improper(w1, n1, d1)
        else:
            n1_imp, d1_imp = random_fraction()
            op1_str = f"{n1_imp}/{d1_imp}"
        
        if op2_type == 'mixed':
            w2, n2, d2 = random_mixed()
            op2_str = f"{w2} {n2}/{d2}"
            n2_imp, d2_imp = to_improper(w2, n2, d2)
        else:
            n2_imp, d2_imp = random_fraction()
            op2_str = f"{n2_imp}/{d2_imp}"
        
        problem = f"{op1_str} {operation} {op2_str}"
        
        # Calculate correct answer
        if operation == '+':
            correct_num = n1_imp * d2_imp + n2_imp * d1_imp
            correct_den = d1_imp * d2_imp
        elif operation == '-':
            correct_num = n1_imp * d2_imp - n2_imp * d1_imp
            correct_den = d1_imp * d2_imp
        elif operation == 'x':
            correct_num = n1_imp * n2_imp
            correct_den = d1_imp * d2_imp
        else:  # division
            correct_num = n1_imp * d2_imp
            correct_den = d1_imp * n2_imp if n2_imp != 0 else 1
        
        # Ensure uniqueness
        key = (problem, n1_imp, d1_imp, n2_imp, d2_imp, operation)
        if key in unique_problems:
            continue
        unique_problems.add(key)
        
        # Simplify correct answer
        correct_frac = Fraction(correct_num, correct_den)
        correct_num_simp = correct_frac.numerator
        correct_den_simp = correct_frac.denominator
        
        # Generate errors for each type
        for label, error_func in error_functions.items():
            if label in ["po-a", "po-b", "a"]:
                student_ans = error_func(correct_num_simp, correct_den_simp)
            else:
                student_ans = error_func(n1_imp, d1_imp, n2_imp, d2_imp, operation)
            
            row = {
                "input": f"Problem: {problem} | Student Answer: {student_ans}",
                "label": label
            }
            data.append(row)
            
            if len(data) >= target_count:
                break
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating improved dataset with more distinctive error patterns...")
    df = generate_improved_dataset(50000)  # 50,000 examples
    df.to_csv("improved_diversed_data.csv", index=False)
    print(f"Generated {len(df)} examples in 'improved_diversed_data.csv'")
    
    # Show distribution
    print("\nLabel distribution:")
    print(df['label'].value_counts()) 