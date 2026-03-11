import torch
from transformers import AutoTokenizer, AutoModel
from sympy import sympify, Rational, simplify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import re

# 1. Define your custom MathBERTClassifier class
class MathBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(MathBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# 2. Load label encoder
df = pd.read_csv("test_set.csv")
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
num_classes = len(le.classes_)

# 3. Load tokenizer and model
model_name = "tbs17/MathBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MathBERTClassifier(model_name, num_classes)
model.load_state_dict(torch.load("best_mathbert_model.pt", map_location=torch.device("cpu")))
model.eval()

# 4. Normalize math expression symbols (÷, ×, x, − → standard)
def normalize_math(expr):
    expr = expr.replace("÷", "/").replace("×", "*").replace("x", "*").replace("−", "-").strip()

    # Only wrap parentheses if the operation is division
    if "/" in expr and "+" not in expr and "-" not in expr and "*" not in expr:
        tokens = expr.split()
        if len(tokens) == 3 and tokens[1] == "/":
            return f"({tokens[0]}) / ({tokens[2]})"

    return expr



# 5. Convert mixed numbers like "3 2/7" → "(3+2/7)"
def convert_mixed_numbers(expr):
    return re.sub(r'(\d+)\s+(\d+)/(\d+)', r'(\1+\2/\3)', expr)

#  Math equivalence check
def is_math_equal(student_ans, problem_expr):
    try:
        problem_expr = convert_mixed_numbers(normalize_math(problem_expr))
        student_ans = convert_mixed_numbers(student_ans)

        correct_value = simplify(sympify(problem_expr))
        student_value = simplify(sympify(student_ans))

        print(f"SymPy check: {problem_expr} = {correct_value}, student = {student_value}")
        return correct_value == student_value
    except Exception as e:
        print(f"SymPy failed: {e}")
        return False

# 6. Prediction function with Top-3 support
def predict_error(problem, student_answer):
    if is_math_equal(student_answer, problem):
        return {
            "predicted_label": "no_error",
            "confidence": 1.0,
            "top_3": [("no_error", 1.0)]
        }

    input_text = f"Problem: {problem} | Student Answer: {student_answer}"
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Top-3 predictions
        top_probs, top_classes = torch.topk(probs, k=3)
        top_labels = le.inverse_transform(top_classes[0].tolist())
        top_values = top_probs[0].tolist()

        return {
            "predicted_label": top_labels[0],
            "confidence": top_values[0],
            "top_3": list(zip(top_labels, top_values))
        }

# 7. Example usage
if __name__ == "__main__":
    print("=== Testing Model on Test Set Examples ===")
    
    # Load test set
    test_df = pd.read_csv("test_set.csv")
    print(f"Total test examples: {len(test_df)}")
    
    # Get unique error labels
    unique_labels = test_df['label'].unique()
    print(f"Error classes found: {unique_labels}")
    
    # Test one example from each error class
    correct_predictions = 0
    total_predictions = 0
    
    print(f"\n=== Testing One Example from Each Error Class ===")
    
    for label in unique_labels:
        # Get one example for this label
        label_example = test_df[test_df['label'] == label].iloc[0]
        
        # Extract problem and student answer from the input text
        input_text = label_example['input']
        
        # Parse the input text to get problem and student answer
        if "Problem: " in input_text and " | Student Answer: " in input_text:
            parts = input_text.split(" | Student Answer: ")
            problem = parts[0].replace("Problem: ", "")
            student_answer = parts[1]
            expected_label = label_example['label']
            
            print(f"\n{'='*50}")
            print(f"Testing Error Class: {expected_label}")
            print(f"{'='*50}")
            
            result = predict_error(problem, student_answer)
            predicted_label = result['predicted_label']
            
            is_correct = predicted_label == expected_label
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            
            print(f"Problem:        {problem}")
            print(f"Student Answer: {student_answer}")
            print(f"Expected:       {expected_label}")
            print(f"Predicted:      {predicted_label} | Confidence: {result['confidence']:.2f}")
            print(f"Status:         {status}")
            print(f"Top 3 guesses:  {result['top_3']}")
            
            # Special check for no_error cases
            if expected_label == "no_error":
                print(f"\n SymPy Check for no_error:")
                print(f"   - Problem: {problem}")
                print(f"   - Student Answer: {student_answer}")
                print(f"   - Should be mathematically equal: {is_math_equal(student_answer, problem)}")
    
    print(f"\n{'='*50}")
    print(f"=== SUMMARY ===")
    print(f"{'='*50}")
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {correct_predictions/total_predictions*100:.1f}%")
    
    # Show breakdown by error class
    print(f"\n=== Performance by Error Class ===")
    for label in unique_labels:
        label_example = test_df[test_df['label'] == label].iloc[0]
        input_text = label_example['input']
        parts = input_text.split(" | Student Answer: ")
        problem = parts[0].replace("Problem: ", "")
        student_answer = parts[1]
        
        result = predict_error(problem, student_answer)
        predicted_label = result['predicted_label']
        is_correct = predicted_label == label
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {label}: Predicted as '{predicted_label}' (Confidence: {result['confidence']:.2f})")
    
    # Test SymPy specifically for no_error cases
    print(f"\n=== Testing SymPy for no_error Detection ===")
    no_error_examples = [
        {"problem": "1/2 + 1/2", "student": "1", "expected": True},
        {"problem": "3/4 - 1/4", "student": "1/2", "expected": True},
        {"problem": "2/3 x 3/2", "student": "1", "expected": True},
        {"problem": "1/2 ÷ 1/4", "student": "2", "expected": True},
        {"problem": "1/2 + 1/3", "student": "5/6", "expected": True},
        {"problem": "1/2 + 1/2", "student": "2", "expected": False},  # Wrong answer
        {"problem": "1/2 x 1/2", "student": "1/4", "expected": True},
    ]
    
    for example in no_error_examples:
        problem = example["problem"]
        student_answer = example["student"]
        expected = example["expected"]
        
        is_equal = is_math_equal(student_answer, problem)
        status = "✅" if is_equal == expected else "❌"
        
        print(f"{status} Problem: {problem} | Student: {student_answer} | SymPy says: {is_equal} | Expected: {expected}")
