# ConsistentAvatar
The code will be uploaded and updated gradually in the near future.
Thanks to [Diffusionrig](https://github.com/adobe-research/diffusion-rig/tree/main) for the excellent work; our code is based on modifications of it.
## Setup & Preparation
## Environment Setup
```bash
conda create -n consistentavatar python=3.8
conda activate consistentavatar
pip install -r requirement.txt
```
## DECA Setup
Before preparing the data for training, make sure to download the DECA source files and checkpoints to set up the environment (you'll need to create an account to access the FLAME resources):
1. Download the pretrained DECA model [deca_model.tar](https://github.com/YadiraF/DECA#:~:text=You%20can%20also%20use%20released%20model%20as%20pretrained%20model%2C%20then%20ignor%20the%20pretrain%20step.)
2. Download FLAME 2020 and extract [generic_model.pkl](https://flame.is.tue.mpg.de/download.php)
3. Download the FLAME texture space and extract [FLAME_texture.npz](https://flame.is.tue.mpg.de/download.php)
4. Download the other files listed below from [DECA's Data Page](https://github.com/YadiraF/DECA/tree/master/data) and put them also in the data/ folder:
```bash
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```
