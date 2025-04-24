"""Define argument parser class."""
import argparse
from pathlib import Path


class ArgParser(object):
    """Argument parser for label.py"""
    def __init__(self):
        """Initialize argument parser."""
        parser = argparse.ArgumentParser()

        # Input report parameters.
        parser.add_argument('--generated_reports_path',
                            default='gn_reports.csv',
                            help='Path to file with generated radiology reports.')
        
        gt_group = parser.add_mutually_exclusive_group()
        gt_group.add_argument('--ground_truth_reports_path',
                            help='Path to file with ground truth radiology reports.')
        gt_group.add_argument('--ground_truth_labels_path',
                            default="gt_labels.csv",
                            help='Path to file with ground truth labels.')
        
        parser.add_argument('--sections_to_extract',
                            nargs='*',
                            default=[],
                            help='Titles of the sections to extract from ' +
                                 'each report.')
        parser.add_argument('--extract_strict',
                            action='store_true',
                            help='Instructs the labeler to only extract the ' +
                                 'sections given by sections_to_extract. ' +
                                 'If this argument is given and a report is ' +
                                 'encountered that does not contain any of ' +
                                 'the provided sections, instead of loading ' +
                                 'the original document, that report will be ' +
                                 'loaded as an empty document.')

        # Phrases
        parser.add_argument('--mention_phrases_dir',
                            default='phrases/mention',
                            help='Directory containing mention phrases for ' +
                                 'each observation.')
        parser.add_argument('--unmention_phrases_dir',
                            default='phrases/unmention',
                            help='Directory containing unmention phrases ' +
                                 'for each observation.')

        # Rules
        parser.add_argument('--pre_negation_uncertainty_path',
                            default='patterns/pre_negation_uncertainty.txt',
                            help='Path to pre-negation uncertainty rules.')
        parser.add_argument('--negation_path',
                            default='patterns/negation.txt',
                            help='Path to negation rules.')
        parser.add_argument('--post_negation_uncertainty_path',
                            default='patterns/post_negation_uncertainty.txt',
                            help='Path to post-negation uncertainty rules.')

        # Output parameters.
        parser.add_argument('--hallucinated_output_path',
                            default='hallucinated_labels.csv',
                            help='Output path to write hallucinated labels to.')
        
        parser.add_argument('--gen_labels_output_path',
                            default='gen_labels.csv',
                            help='Output path to write labels from generated reports to.')
        
        parser.add_argument('--gt_labels_output_path',
                            default='gt_binary_labels.csv',
                            help='Output path to write ground truth labels to.')

        # Misc.
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Print progress to stdout.')

        self.parser = parser

    def parse_args(self):
        """Parse and validate the supplied arguments."""
        args = self.parser.parse_args()

        # Convert paths, handling None values
        args.generated_reports_path = Path(args.generated_reports_path)
        if args.ground_truth_reports_path is not None:
            args.ground_truth_reports_path = Path(args.ground_truth_reports_path)
        if args.ground_truth_labels_path is not None:
            args.ground_truth_labels_path = Path(args.ground_truth_labels_path)
        args.mention_phrases_dir = Path(args.mention_phrases_dir)
        args.unmention_phrases_dir = Path(args.unmention_phrases_dir)
        args.hallucinated_output_path = Path(args.hallucinated_output_path)
        args.gen_labels_output_path = Path(args.gen_labels_output_path)
        args.gt_labels_output_path = Path(args.gt_labels_output_path)
        return args
