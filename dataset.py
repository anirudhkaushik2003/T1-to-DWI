import nibabel as nib
import numpy as np
import os
import glob
import cv2
from torch.utils.data import Dataset


from tqdm import tqdm
total_failed_t1 = 0
total_failed_dwi = 0

shapes = {}
def get_data():
    t1_list = []

    for file in tqdm(glob.glob("/ssd_scratch/cvit/anirudhkaushik/data/t1/*"), desc="Loading T1 data"):
        try:
            data = nib.load(file).get_fdata()
            data = np.array(data)

            try:
                shapes[data.shape[2]] += 1
            except:
                shapes[data.shape[2]] = 1
            t1_list.append(data)
        except:
            pass

    print(shapes)
    return t1_list

class t1Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0

if __name__ == "__main__":
    get_data()





