import pandas as pd
from args import ArgParser
from labeler import MedLabeler
from chair_metrics import ChairMetrics
from constants import CATEGORIES

def save_results(output_path, labels):
    """Save labels and positions to a CSV file."""
    df = pd.DataFrame()
    for i, cat in enumerate(CATEGORIES):
        df[cat] = labels[:,i]
    df.to_csv(output_path, index=False)

def save_positions(output_path, positions):
    """Save positions to a CSV file with proper column headers for each category."""
    df = pd.DataFrame()
    # Create a column for each category's position
    for i, cat in enumerate(CATEGORIES):
        df[cat] = positions[:,i]
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    args = ArgParser().parse_args()
    
    labeler = MedLabeler(args)
    metrics = ChairMetrics()
    
    gt_labels, gen_labels, gt_positions, gen_positions = labeler.get_labels_and_positions()
    
    chair_s, chair_i, hall_labels, non_hall_labels, gt_binary, gen_binary = metrics.compute_chair_metrics(gt_labels, gen_labels)
    
    micro_f1, macro_f1 = metrics.compute_f1(gt_binary, gen_binary)
    
    save_results(args.hallucinated_output_path, hall_labels)
    save_results("non_hallucinated_labels.csv", non_hall_labels)
    save_results(args.gen_labels_output_path, gen_labels)
    save_results("gen_binary_labels.csv", gen_binary)
    save_results(args.gt_labels_output_path, gt_binary)
    
    save_positions("generated_positions.csv", gen_positions)
    
    print(f"Chair I: {chair_i}, Chair S: {chair_s}, Micro F1: {micro_f1}, Macro F1: {macro_f1}")
    with open("metrics.txt", "w") as f:
        f.write(f"Chair I: {chair_i}, Chair S: {chair_s}, Micro F1: {micro_f1}, Macro F1: {macro_f1}")
    