import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

MIMIC_CXR_IMG_DIR = "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/"
MIMIC_CXR_DIR = "/datasets/mimic-cxr/physionet.org/files/mimic-cxr/2.0.0/"

class MIMIC_CXR_Dataset(Dataset):
    def __init__(self, csv_path: str = "data/mimic-cxr/cxr-study-list-with-ap-image-path.csv"):
        self.df = pd.read_csv(csv_path)
        # Standard transforms for BioViL-T
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load and transform image
        img_path = MIMIC_CXR_IMG_DIR + row["ap_image_path"]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load report
        with open(MIMIC_CXR_DIR + row["report_path"], "r") as f:
            report = f.read()

        return image, img_path, report