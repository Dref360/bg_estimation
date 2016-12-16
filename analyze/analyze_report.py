from pathlib import Path

import numpy as np
import pandas as pd

head = ["VIDNAME", "AGE", "pEPs", "pCEPs", "MSSSIM", "PSNR", 'CQM']
inner_h = ["AGE", "pEPs", "pCEPs", "MSSSIM", "PSNR"]

vgg = pd.read_csv("../../output_rep/reportvgg.csv")
unet = pd.read_csv("../../output_rep/reportunet.csv")
vgg_30 = pd.read_csv("../../reportvgg_30.csv")
unet_30 = pd.read_csv("../../reportunet_30.csv")
sf_100 = pd.read_csv("../../reportsf_100.csv")


def get_info(vgg):
    vgg2 = vgg.groupby("VIDNAME")
    vgg3 = vgg2[["AGE", "pEPs", "pCEPs", "MSSSIM", "PSNR"]].mean()
    vgg3["VIDNAME"] = vgg3.index
    vgg3["CAT"] = vgg3.apply(lambda x: str(Path(x["VIDNAME"]).parent.parent).split('/')[-1], 1)
    vgg4 = vgg3.groupby("CAT")[["AGE", "pEPs", "pCEPs", "MSSSIM", "PSNR"]].agg([np.mean, np.std])
    return vgg4


mean_vgg = get_info(vgg)
mean_unet = get_info(unet)
mean_unet_30 = get_info(unet_30)
mean_vgg_30 = get_info(vgg_30)
mean_sf_100 = get_info(sf_100)


mean_vgg.to_csv("mean_vgg.csv")
mean_unet.to_csv("mean_unet.csv")
mean_sf_100.to_csv("mean_sf_100.csv")

print("VGG")
print(mean_vgg)
print()
print("VGG+")
print(mean_unet)
print("VGG 30")
print(mean_vgg_30)
print()
print("VGG+ 30")
print(mean_unet_30)
print("SF 100")
print(mean_sf_100)
