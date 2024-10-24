
import os
import re
import cv2
import glob
import yaml
import torch
import pydicom
import numpy as np
import albumentations as A
import torch.utils.data
from tqdm import tqdm

from Encoders import ImageEncoder, TextEncoder


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, config, transforms):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive file names
        """

        self.config = config
        self.transforms = transforms
        self.device = config['device']
        self.folder_path = config['folder_path']
        self.batch_size = config['batch_size']
        self.batch_size = 4

        self.image_encoder = ImageEncoder(config).to(self.device)
        self.text_encoder = TextEncoder(config).to(self.device)

        self.image_encoder.eval()
        self.text_encoder.eval()

        self.data = self.get_data()     ## 36681 pairs of size (1, 2048) and (1, 768)
        self.train_data, self.test_data = torch.utils.data.random_split(self.data, [0.8, 0.2])
        
    
    def get_data(self):
        report_mri_pairs = []
        reports_paths = self.get_all_txt_files(self.folder_path)

        for report in tqdm(reports_paths[:100]):
            
            # Get report label and encode it
            label = self.file_labelling(report)
            with torch.no_grad():
                encoded_label = self.text_encoder(label)

            # Get MRI images path for current report
            report_folder = os.path.splitext(report)[0]
            mri_paths = glob.glob(os.path.join(report_folder, '*.dcm'))
            
            # Create a tuple (mri, report label) for each MRI
            for mri_path in mri_paths:
                mri_image = self.load_dicom(mri_path)                           ## (512, 512, 3) shape, 0-255 pixel values
                mri_image = torch.tensor(mri_image).to(self.device)             ## Convert to tensor and move to device
                mri_image = mri_image.permute(2, 0, 1).float().unsqueeze(0)     ## Change to (1, 3, 512, 512) shape for encoder
                
                with torch.no_grad():
                    encoded_image = self.image_encoder(mri_image)  ## (1, 2048) shape

                # Store tensors on CPU to save GPU memory
                report_mri_pairs.append(encoded_image.cpu(), encoded_label.cpu())

        return report_mri_pairs            

    def get_all_txt_files(self, folder_path):
        return glob.glob(os.path.join(folder_path, '**/*.txt'), recursive=True)

    def file_labelling(self, file_path):
        with open(file_path, 'r') as file:
            full_text = file.read()
            file.seek(0)                                    ## Reset file pointer to the beginning
            report_sections = self.get_file_sections(file)  ## Get sections of the report
            label_type, label = self.get_file_label(report_sections, full_text) 

        return label
    
    def get_file_sections(self, file):
        report_text = file.read()
        report_dict = {}
        
        current_section = None
        lines = report_text.strip().split('\n')     ## Split the report into lines and process
        section_pattern = re.compile(r'^[A-Z ]+:')  ## Regex pattern to identify sections
        
        for line in lines:
            line = line.strip() 
            match = section_pattern.match(line)

            if match:
                current_section = match.group(0)[:-1].strip()   ## Get section name and strip whitespace
                report_dict[current_section] = []
                if len(line) > match.span()[1] + 1:
                    report_dict[current_section].append(line[match.span()[1]:])
                    
            elif current_section:   ## If section name is found, append line to current section
                report_dict[current_section].append(line)
        
        # Join lines in each section to form a single string
        for key in report_dict:
            report_dict[key] = ' '.join(report_dict[key]).strip()
        
        return report_dict
    
    def get_file_label(self, report_dict, full_text):
        if 'IMPRESSION' in report_dict:
            return 'IMPRESSION', report_dict['IMPRESSION']
        elif 'FINDINGS' in report_dict:
            return 'FINDINGS', report_dict['FINDINGS']
        elif 'REASON FOR EXAMINATION' in report_dict:
            return 'REASON FOR EXAMINATION', report_dict['REASON FOR EXAMINATION']
        elif 'REASON FOR EXAM' in report_dict:
            return 'REASON FOR EXAM', report_dict['REASON FOR EXAM']
        else:
            return 'ENTIRE FILE', full_text.strip()

    def load_dicom(self, path):
        dicom = pydicom.dcmread(path)               ## Read the DICOM file
        image = dicom.pixel_array                   ## Convert to numpy array, shape: (3056, 2544), pixel values 0 -3636
        image = image.astype(np.float32)            ## Convert to float32
        image = np.stack([image] * 3, axis=-1)      ## Convert to 3-channel (RGB-like)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)    ## Resize to 512x512
        image = (image - image.min()) / (image.max() - image.min())            ## Normalize to 0-1

        return image
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def __len__(self):
        return len(self.data) // self.batch_size
    
    def get_transforms(self, mode="train"):
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(self.config.size, self.config.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.config.size, self.config.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
    
if __name__ == "__main__": 
    config = yaml.safe_load(open("./config.yaml"))
    config["device"] = 'cuda:3' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'

    dataset = CLIPDataset(config, None)
    image, label = dataset[0]
    print(image.shape, label.shape)