import ast


      
def _safe_divide(n, d):
  return n / d if d else 0


def pipeline_report(true, predicted):
    
    """
    Objective: Compute precision, recall, F1 and Accuracy of gratitude pipeline
    Inputs:
        - true: list of annotations
        - predicted: list of predicted labels
    Outputs:
        - metrics, dict: metrics of evalutation
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(len(true)):
        if true[i] == predicted[i] and true[i] == 1:
            true_positives += 1
        elif true[i] == predicted[i] and true[i] == 0:
            true_negatives += 1
        elif true[i] != predicted[i] and predicted[i] == 1:
            false_positives += 1
        else:
            false_negatives += 1 
        
    recall = _safe_divide(true_positives, (true_positives+false_negatives))
    precision = _safe_divide(true_positives, (true_positives+false_positives))
    F1 = _safe_divide(2*recall*precision, (precision+recall))
    Accuracy = _safe_divide(true_positives, (true_positives + false_positives + false_negatives))

    
    metrics = {
        "true positives": true_positives,
        "true negatives": true_negatives,
        "false positives": false_positives,
        "false negatives": false_negatives,
        "recall": recall,
        "precision": precision,
        "f1": F1,
        "accuracy": Accuracy
        }
    return metrics





