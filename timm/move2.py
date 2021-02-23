import shutil
import os
from tqdm import tqdm
cp_list = os.listdir('work_dirs2')
for cp in tqdm(cp_list):
    shutil.copy(f'work_dirs2/{cp}/model_50.pth',f'quanzhong2/{cp}.pth')