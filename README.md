# ConsistentAvatar
The code will be uploaded and updated gradually in the near future.

## Setup & Preparation
## Environment Setup
conda create -n consistentavatar python=3.8 <br>
conda activate consistentavatar <br>
pip install -r requirement.txt <br>

## DECA Setup
Before preparing the data for training, make sure to download the DECA source files and checkpoints to set up the environment (you'll need to create an account to access the FLAME resources):
1. Download the pretrained DECA model [deca_model.tar](https://github.com/YadiraF/DECA#:~:text=You%20can%20also%20use%20released%20model%20as%20pretrained%20model%2C%20then%20ignor%20the%20pretrain%20step.)
2. Download FLAME 2020 and extract [generic_model.pkl](https://flame.is.tue.mpg.de/download.php)
3. Download the FLAME texture space and extract [FLAME_texture.npz](https://flame.is.tue.mpg.de/download.php)
4. Download the other files listed below from [DECA's Data Page](https://github.com/YadiraF/DECA/tree/master/data) and put them also in the data/ folder:<br>
`
data/ \\
  deca_model.tar \\
  generic_model.pkl<br>
  FLAME_texture.npz<br>
  fixed_displacement_256.npy <br>
  head_template.obj <br>
  landmark_embedding.npy <br>
  mean_texture.jpg <br>
  texture_data_256.npy <br>
  uv_face_eye_mask.png <br>
  uv_face_mask.png <br>
`
