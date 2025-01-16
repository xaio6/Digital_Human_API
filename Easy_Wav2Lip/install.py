version = 'v8.3'

import os
import re
import argparse
import shutil
import subprocess
from IPython.display import clear_output

from easy_functions import (format_time,
                            load_file_from_url,
                            load_model,
                            load_predictor)
                            # Get the location of the basicsr package
import os
import shutil
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

# Get the location of the basicsr package
def get_basicsr_location():
    result = subprocess.run(['pip', 'show', 'basicsr'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Location: ' in line:
            return line.split('Location: ')[1]
    return None

# Move and replace a file to the basicsr location
def move_and_replace_file_to_basicsr(file_name):
    basicsr_location = get_basicsr_location()
    if basicsr_location:
        destination = os.path.join(basicsr_location, file_name)
        # Move and replace the file
        shutil.copyfile(file_name, destination)
        print(f'File replaced at {destination}')
    else:
        print('Could not find basicsr location.')

# Example usage
file_to_replace = 'degradations.py'  # Replace with your file name
move_and_replace_file_to_basicsr(file_to_replace)


from enhance import load_sr

working_directory = os.getcwd()

# download and initialize both wav2lip models
print("downloading wav2lip essentials")
load_file_from_url(
    url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
    model_dir="checkpoints",
    progress=True,
    file_name="Wav2Lip_GAN.pth",
)
model = load_model(os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth"))
print("wav2lip_gan loaded")
load_file_from_url(
    url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip.pth",
    model_dir="checkpoints",
    progress=True,
    file_name="Wav2Lip.pth",
)
model = load_model(os.path.join(working_directory, "checkpoints", "Wav2Lip.pth"))
print("wav2lip loaded")

# download gfpgan files
print("downloading gfpgan essentials")
load_file_from_url(
    url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/GFPGANv1.4.pth",
    model_dir="checkpoints",
    progress=True,
    file_name="GFPGANv1.4.pth",
)
load_sr()

# load face detectors
print("initializing face detectors")
load_file_from_url(
    url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/shape_predictor_68_face_landmarks_GTX.dat",
    model_dir="checkpoints",
    progress=True,
    file_name="shape_predictor_68_face_landmarks_GTX.dat",
)

load_predictor()

# write a file to signify setup is done
with open("installed.txt", "w") as f:
    f.write(version)
print("Installation complete!")
print(
    "If you just updated from v8 - make sure to download the updated Easy-Wav2Lip.bat too!"
)
