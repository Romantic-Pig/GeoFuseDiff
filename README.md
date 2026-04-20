# 🌍 GeoFuseDiff: Geographic Fusion-Driven Diffusion Model for Meteorological Downscaling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

GeoFuseDiff/
├── DatasetUS.py      
├── Network.py         
├── TrainDiffusion.py  
├── Inference.py      
├── README.md

We recommend using Conda to create a virtual environment and installing the following dependencies:
conda create -n geofusediff python=3.9
conda activate geofusediff

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install rasterio numpy scipy matplotlib tqdm tensorboard


Please organize your .tif raster data according to the following directory structure. Filenames must be in the format YYYYMMDDHH.tif (e.g., 2019010100.tif) so that DatasetUS.py can automatically parse the time information (month and daytime):

data_dir/
├── ERA5_2m/                 # Coarse Resolution
├── CLDAS_2m/                # Fine Resolution - training)
├── DEM/
│   └── dem_1km.tif          
└── Land_use/
    └── LinYi/              
