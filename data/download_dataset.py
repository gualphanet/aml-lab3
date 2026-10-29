import subprocess
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP = "tiny-imagenet-200.zip"
DEST = "dataset/tiny-imagenet"

# Scarica solo se il file non esiste già
if not Path(ZIP).exists():
    subprocess.run(["wget", "-q", "--show-progress", URL], check=True)

# Decomprimi in modalità silenziosa (-q), altrimenti Colab stampa migliaia di righe
subprocess.run(["unzip", "-q", ZIP, "-d", DEST], check=True)

print("Fatto. Contenuto:", list(Path(DEST).iterdir()))


# We need to adjust the format of the val split of the dataset to be used with ImageFolder.
import os
import shutil

with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')