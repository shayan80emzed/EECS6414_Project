"""Define mention aggregator class."""
import numpy as np
from tqdm import tqdm

from negbio.chexpert.constants import NEGATIVE, UNCERTAIN, POSITIVE, SUPPORT_DEVICES, NO_FINDING, OBSERVATION, \
    NEGATION, UNCERTAINTY, CARDIOMEGALY


class Aggregator(object):
    """Aggregate mentions of observations from radiology reports."""

    def __init__(self, categories, verbose=False):
        self.categories = categories
        self.verbose = verbose

    def dict_to_vec(self, d, pos_dict=None):
        """
        Convert a dictionary of the form

        {cardiomegaly: [1],
         opacity: [u, 1],
         fracture: [0]}

        into vectors of the form

        [np.nan, np.nan, 1, u, np.nan, ..., 0, np.nan]
        """
        vec = []
        pos_vec = []
        for category in self.categories:
            # There was a mention of the category.
            if category in d:
                label_list = d[category]
                # Only one label, no conflicts.
                if len(label_list) == 1:
                    vec.append(label_list[0])
                # Multiple labels.
                else:
                    # Case 1. There is negated and uncertain.
                    if NEGATIVE in label_list and UNCERTAIN in label_list:
                        vec.append(UNCERTAIN)
                    # Case 2. There is negated and positive.
                    elif NEGATIVE in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 3. There is uncertain and positive.
                    elif UNCERTAIN in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 4. All labels are the same.
                    else:
                        vec.append(label_list[0])
                
                # Add position if available
                if pos_dict and category in pos_dict:
                    pos_vec.append(pos_dict[category])
                else:
                    pos_vec.append(-1)  # Use -1 to indicate no position found

            # No mention of the category
            else:
                vec.append(np.nan)
                pos_vec.append(-1)

        return vec, pos_vec

    def aggregate(self, collection):
        labels = []
        positions = []
        documents = collection.documents
        if self.verbose:
            print("Aggregating mentions...")
            documents = tqdm(documents)
        for document in documents:
            label_dict = {}
            pos_dict = {}
            impression_passage = document.passages[0]
            no_finding = True
            for annotation in impression_passage.annotations:
                category = annotation.infons[OBSERVATION]

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # If at least one non-support category has a uncertain or
                # positive label, there was a finding
                if (category != SUPPORT_DEVICES and
                        label in [UNCERTAIN, POSITIVE]):
                    no_finding = False

                # Don't add any labels for No Finding
                if category == NO_FINDING:
                    continue

                # add exception for 'chf' and 'heart failure'
                if ((label in [UNCERTAIN, POSITIVE]) and
                        (annotation.text == 'chf' or
                         annotation.text == 'heart failure')):
                    if CARDIOMEGALY not in label_dict:
                        label_dict[CARDIOMEGALY] = [UNCERTAIN]
                        pos_dict[CARDIOMEGALY] = annotation.locations[0].offset
                    else:
                        label_dict[CARDIOMEGALY].append(UNCERTAIN)

                if category not in label_dict:
                    label_dict[category] = [label]
                    pos_dict[category] = annotation.locations[0].offset
                else:
                    label_dict[category].append(label)

            if no_finding:
                label_dict[NO_FINDING] = [POSITIVE]
                pos_dict[NO_FINDING] = 0  # Position 0 for No Finding

            label_vec, pos_vec = self.dict_to_vec(label_dict, pos_dict)
            labels.append(label_vec)
            positions.append(pos_vec)

        return np.array(labels), np.array(positions)


class NegBioAggregator(Aggregator):
    LABEL_MAP = {UNCERTAIN: 'Uncertain', POSITIVE: 'Positive', NEGATIVE: 'Negative'}

    def aggregate_doc(self, document):
        """
        Aggregate mentions of observations from radiology reports.

        Args:
            document (BioCDocument):

        Returns:
            BioCDocument
        """
        label_dict = {}
        pos_dict = {}
        no_finding = True
        for passage in document.passages:
            for annotation in passage.annotations:
                category = annotation.infons[OBSERVATION]

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # If at least one non-support category has a uncertain or
                # positive label, there was a finding
                if category != SUPPORT_DEVICES \
                        and label in [UNCERTAIN, POSITIVE]:
                    no_finding = False

                # Don't add any labels for No Finding
                if category == NO_FINDING:
                    continue

                # add exception for 'chf' and 'heart failure'
                if label in [UNCERTAIN, POSITIVE] \
                        and (annotation.text == 'chf' or annotation.text == 'heart failure'):
                    if CARDIOMEGALY not in label_dict:
                        label_dict[CARDIOMEGALY] = [UNCERTAIN]
                        pos_dict[CARDIOMEGALY] = annotation.locations[0].offset
                    else:
                        label_dict[CARDIOMEGALY].append(UNCERTAIN)

                if category not in label_dict:
                    label_dict[category] = [label]
                    pos_dict[category] = annotation.locations[0].offset
                else:
                    label_dict[category].append(label)

        if no_finding:
            label_dict[NO_FINDING] = [POSITIVE]
            pos_dict[NO_FINDING] = 0  # Position 0 for No Finding

        for category in self.categories:
            key = 'CheXpert/{}'.format(category)
            # There was a mention of the category.
            if category in label_dict:
                label_list = label_dict[category]
                # Only one label, no conflicts.
                if len(label_list) == 1:
                    document.infons[key] = self.LABEL_MAP[label_list[0]]
                # Multiple labels.
                else:
                    # Case 1. There is negated and uncertain.
                    if NEGATIVE in label_list and UNCERTAIN in label_list:
                        document.infons[key] = self.LABEL_MAP[UNCERTAIN]
                    # Case 2. There is negated and positive.
                    elif NEGATIVE in label_list and POSITIVE in label_list:
                        document.infons[key] = self.LABEL_MAP[POSITIVE]
                    # Case 3. There is uncertain and positive.
                    elif UNCERTAIN in label_list and POSITIVE in label_list:
                        document.infons[key] = self.LABEL_MAP[POSITIVE]
                    # Case 4. All labels are the same.
                    else:
                        document.infons[key] = self.LABEL_MAP[label_list[0]]

            # No mention of the category
            else:
                pass
        return document
