import pandas as pd
import os
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *

class MedLabeler:
    def __init__(self, args):
        self.args = args
        self.args.reports_path = "reports_path.csv"
        
        self.loader = Loader(self.args.reports_path,
                        self.args.sections_to_extract,
                        self.args.extract_strict)

        self.extractor = Extractor(self.args.mention_phrases_dir,
                            self.args.unmention_phrases_dir,
                            verbose=self.args.verbose)
        
        self.classifier = Classifier(self.args.pre_negation_uncertainty_path,
                                self.args.negation_path,
                                self.args.post_negation_uncertainty_path,
                                verbose=self.args.verbose)
        
        self.aggregator = Aggregator(CATEGORIES,
                                verbose=self.args.verbose)

    def _prepare_reports(self):
        """Prepare and concatenate reports for processing."""
        if self.args.ground_truth_labels_path is None:
            gt_size = len(pd.read_csv(self.args.ground_truth_reports_path, header=None))
            concat_reports = open(self.args.ground_truth_reports_path).read() + '\n' + open(self.args.generated_reports_path).read()
        else:
            concat_reports = open(self.args.generated_reports_path).read()
            gt_size = None

        with open(self.args.reports_path, 'w') as f: 
            f.write(concat_reports)
        
        return gt_size

    def _process_reports(self):
        """Process reports through the pipeline to get labels and positions."""
        self.loader.load()
        self.extractor.extract(self.loader.collection)
        self.classifier.classify(self.loader.collection)
        labels, positions = self.aggregator.aggregate(self.loader.collection)
        
        os.remove(self.args.reports_path)
        return labels, positions

    def get_labels_and_positions(self):
        """Get labels and positions for both ground truth and generated reports."""
        gt_size = self._prepare_reports()
        labels, positions = self._process_reports()
        
        if self.args.ground_truth_labels_path is None:
            gt_labels = labels[:gt_size,:]
            gen_labels = labels[gt_size:,:]
            gt_positions = positions[:gt_size,:]
            gen_positions = positions[gt_size:,:]
        else:
            gt_labels = pd.read_csv(self.args.ground_truth_labels_path).to_numpy()
            gen_labels = labels
            gt_positions = None
            gen_positions = positions
            
        return gt_labels, gen_labels, gt_positions, gen_positions 