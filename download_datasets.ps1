# Run from project root:
# powershell -ExecutionPolicy Bypass -File download_datasets.ps1

Write-Host "Creating folders..."

New-Item -ItemType Directory -Force -Path "data/raw/wikiart" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/artbench" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/places365" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/coco_subset" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null

Write-Host "Installing Kaggle CLI if needed..."
python -m pip install --upgrade kaggle

Write-Host "Downloading WikiArt from Kaggle..."
kaggle datasets download -d steubk/wikiart -p data/raw/wikiart --unzip

Write-Host "Downloading ArtBench from official source..."
curl.exe -L -o data/raw/artbench/artbench-10-imagefolder-split.tar "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar"

Write-Host "Extracting ArtBench..."
tar -xf data/raw/artbench/artbench-10-imagefolder-split.tar -C data/raw/artbench/

Write-Host "Downloading Places365 validation 256..."
curl.exe -L -o data/raw/places365/val_256.tar "http://data.csail.mit.edu/places/places365/val_256.tar"

Write-Host "Extracting Places365..."
tar -xf data/raw/places365/val_256.tar -C data/raw/places365/

Write-Host "Downloading COCO train2017..."
curl.exe -L -o data/raw/coco_subset/train2017.zip "http://images.cocodataset.org/zips/train2017.zip"

Write-Host "Extracting COCO train2017..."
tar -xf data/raw/coco_subset/train2017.zip -C data/raw/coco_subset/

Write-Host "Done."