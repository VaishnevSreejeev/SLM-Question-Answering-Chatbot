from sklearn.metrics import f1_score

def calculate_em(prediction, ground_truth):
    """
    Calculates Exact Match (EM) score.
    """
    return prediction.strip() == ground_truth.strip()

def calculate_f1(prediction, ground_truth):
    """
    Calculates F1 score.
    """
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1