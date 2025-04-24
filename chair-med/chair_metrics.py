from sklearn.metrics import f1_score

class ChairMetrics:
    def __init__(self):
        pass

    def compute_chair_metrics(self, gt_labels, gen_labels):
        """Compute CHAIR metrics for hallucination detection."""
        # Convert to binary labels (1 or -1 -> True, 0 -> False)
        gt_binary = (gt_labels == 1.0) | (gt_labels == -1.0)
        gen_binary = (gen_labels == 1.0) | (gen_labels == -1.0)
        
        # Compute hallucination labels
        hall_labels = (gen_binary & ~gt_binary)
        non_hall_labels = (gen_binary & gt_binary)
        
        # Compute CHAIR metrics
        chair_s = (hall_labels.sum(axis=1) > 0).sum() / len(hall_labels)
        chair_i = hall_labels.sum() / gen_binary.sum()
        
        return chair_s, chair_i, hall_labels, non_hall_labels, gt_binary, gen_binary

    def compute_f1(self, gt_binary, gen_binary):
        """Compute F1 scores for label prediction."""
        micro_f1 = f1_score(gt_binary, gen_binary, average='micro')
        macro_f1 = f1_score(gt_binary, gen_binary, average='macro')
        return micro_f1, macro_f1
