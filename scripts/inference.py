import argparse
import os
import sys
import json
from torchvision import utils as vutils
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from glob import glob

from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


from torchvision.utils import save_image

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset

import pickle
def load():
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)
    return deca

def create_inter_data(deca, dataset, modes, meanshape_path="",target=""):
    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    #print(img2.shape)
    with th.no_grad():
        code2 = deca.encode(img2)
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")
    start_time = time.time()
    print(len(dataset))
    for i in range(len(dataset) - 1):
        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")
        with th.no_grad():
            code1 = deca.encode(img1)
          
        # To align the face when the pose is changing
        ffhq_center = None
        #ffhq_center = deca.decode(code1, return_ffhq_center=True)

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = th.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        for mode in modes:

            code = {}
            for k in code1:
                code[k] = code1[k].clone()

            origin_rendered = None

            if mode == "pose":
                code["pose"][:, :3] = code2["pose"][:, :3]
            elif mode == "light":
                code["light"] = code2["light"]
            elif mode == "exp":
                code["exp"] = code2["exp"]
                code["pose"][:, 3:] = code2["pose"][:, 3:]
            elif mode == "exp_pose":
                code["exp"] = code2["exp"]
            elif mode == "latent":
                pass
            #print(original_image.shape)
            #image1 = TF.to_pil_image(original_image.squeeze())
            #image1.save("original.png")
            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )

            origin_rendered = opdict["rendered_images"].detach()

            batch = {}
            batch["image"] = original_image * 2 - 1
            batch["image2"] = image2 * 2 - 1
            batch["rendered"] = opdict["rendered_images"].detach()
            batch["normal"] = opdict["normal_images"].detach()
            batch["albedo"] = opdict["albedo_images"].detach()
            batch["mode"] = mode
            batch["origin_rendered"] = origin_rendered
            #vutils.save_image(batch["rendered"], './data/rendered/111.png' , normalize=True)
            #vutils.save_image(batch["origin_rendered"], './data/rendered/222.png' , normalize=True)
            #vutils.save_image(normal, './data/normal/' + os.path.split(args.target)[1], normalize=True)
            #vutils.save_image(albedo, './data/albedo/' + os.path.split(args.target)[1], normalize=True)
            yield batch


def main():
    args = create_argparser().parse_args()
    deca = load()
    print("have loaded deca...")
    #--source data/biden_aligned/baiden.png --target data/obama_aligned/Obama.png --output_dir results --modes exp --model_path log/stage2/model005000.pt --meanshape personal_deca.lmdb/mean_shape.pkl --timestep_respacing ddim20
    args.output_dir = "Input your save dir"
    args.timestep_respacing = "ddim20"
  
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
          **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    args.model_path = "log/ID1-5-stage2/model005000.pt"
    ckpt = th.load(args.model_path)
  
    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()
    for i in range(0,350):
      args.source = "Input your source image"
      args.target = "Input your target image"
      args.modes = "exp_pose"
      imagepath_list = []
  
      if not os.path.exists(args.source) or not os.path.exists(args.target):
          print("source file or target file doesn't exists.")
          return
  
      imagepath_list = []
      if os.path.isdir(args.source):
          imagepath_list += (
              glob(args.source + "/*.jpg")
              + glob(args.source + "/*.png")
              + glob(args.source + "/*.bmp")
          )
      else:
          imagepath_list += [args.source]
      imagepath_list += [args.target]
      #print(imagepath_list)
      #args.image_size = 512
      #print("xxxx")
      #print(os.path.split(args.target)[1])
      dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)
  
      modes = args.modes.split(",")
  
      data = create_inter_data(deca,dataset, modes, args.meanshape,args.target)
  
      sample_fn = (
          diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
      )
  
      os.system("mkdir -p " + args.output_dir)
  
      th.manual_seed(81)
      noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")
  
      vis_dir = args.output_dir
  
      idx = 0
      transform = transforms.Compose([
          transforms.Resize((256,256)),
          transforms.ToTensor(),
      ])
      for batch in data:
          start = time.time()
          image = batch["image"]
          image2 = batch["image2"]
         
          tensor_to_pil = transforms.ToPILImage()
      
         
          tsd_path = 'your tsd dir/' + os.path.split(args.target)[1].replace('jpg','png')
          tsd = transform(Image.open(mask_path)).unsqueeze(0).to('cuda')
          rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]
          physic_cond = th.cat([tsd, normal, albedo], dim=1)
  
          image = image
          physic_cond = physic_cond
          
          with th.no_grad():
              if batch["mode"] == "latent":
                  detail_cond = model.encode_cond(image2)
              else:
                  detail_cond = model.encode_cond(image)
  
          sample = sample_fn(
              model,
              (1, 3, args.image_size, args.image_size),
              noise=noise,
              clip_denoised=args.clip_denoised,
              model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond},
          )
          #save_image(
              #sample, os.path.join(vis_dir, "{}_333".format(idx) + batch["mode"]) + args.target.split('/')[-1]
          #)
          sample = (sample + 1) / 2.0
          sample = sample.contiguous()
          print(time.time()-start)
          save_image(
              sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + args.target.split('/')[-1]
          )
          
          idx += 1
        

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        source="",
        target="",
        output_dir="",
        modes="pose,exp,light",
        meanshape="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

