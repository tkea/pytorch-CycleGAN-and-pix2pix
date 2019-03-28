"""
Downloads GTA images and saves them in the following folders:
data/gta/input: Images from GTAV_SATELLITE_8192x8192.png
data/gta/target1: Images from GTAV_ATLUS_8192x8192.png
data/gta/target2: Images from GTAV_ROADMAP_8192x8192.png


"""

# OPTIONS
DOWNLOAD_IMAGES = True
# Remove images with only grass or water
FILTER_DATA = True
# Enable rotation augmentation
AUGMENT_DATA = True
# Save images Height and width
M, N = 256, 256
# Percentage validation size
val_size = 10
USE_ROADMAP_AS_TARGET=True

from PIL import Image
import requests
from io import BytesIO
import os
import shutil
import errno
import numpy as np
import matplotlib.pyplot as plt
import requests
def download_url(url, target_path):
    if os.path.isfile(target_path):
        print("File already downloaded.", target_path)
        return
    print("Downloading image:",url , "Saving to:", target_path)
    img_data = requests.get(url).content
    filedir = os.path.dirname(target_path)
    os.makedirs(filedir, exist_ok=True)
    with open(target_path, 'wb') as handler:
        handler.write(img_data)

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gta_data")
#target_dir = os.path.join(save_path, "trainB")
#input_dir = os.path.join(save_path, "trainA")
original_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"gta_images")


ORIGINAL_IMAGE_URLS = [
    "http://blog.damonpollard.com/wp-content/uploads/2013/09/GTAV_SATELLITE_8192x8192.png",
    "http://blog.damonpollard.com/wp-content/uploads/2013/09/GTAV_ROADMAP_8192x8192.png",
    "http://blog.damonpollard.com/wp-content/uploads/2013/09/GTAV_ATLUS_8192x8192.png"
]

if DOWNLOAD_IMAGES:
    for url in ORIGINAL_IMAGE_URLS:
        
        imname = url.split("/")[-1]
        target_path = os.path.join(original_image_path, imname)
        
        download_url(url, target_path)

if os.path.isdir(save_path):
    answer = input("Folder {} already exist. Are you sure you want to overrwrite it? [y/n]".format(save_path)).lower()
    if answer == 'y' or answer == 'yes' or answer == '1':
        print("Removing old content...")
        shutil.rmtree(save_path)
    else:
        print("Cancelling...")
        exit(1)
print("Can't find gta dataset, making dataset")
os.makedirs(os.path.join(save_path, "train"), exist_ok=True)
os.makedirs(os.path.join(save_path, "val"), exist_ok=True)

def save_aligned(im1, im2, path):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)
    im = np.concatenate((im1, im2), axis=1)
    im = Image.fromarray(im)
    im.save(path)


images = []
IMAGE_NAMES = [
    "GTAV_ROADMAP_8192x8192.png",
    "GTAV_ATLUS_8192x8192.png",
    "GTAV_SATELLITE_8192x8192.png"
]
for image_name in IMAGE_NAMES:
    path = os.path.join(original_image_path, image_name)
    images.append(np.array(Image.open(path)))

print("Chopping them up into {}x{} images".format(M,N))
im0, im1, im2 = images
parted0 = [im0[x: x + M, y: y + N] for x in range(0, im0.shape[0], M) for y in range(0, im0.shape[1], N)]
parted1 = [im1[x: x + M, y: y + N] for x in range(0, im1.shape[0], M) for y in range(0, im1.shape[1], N)]
parted2 = [im2[x: x + M, y: y + N] for x in range(0, im2.shape[0], M) for y in range(0, im2.shape[1], N)]
idx = list(range(len(parted0)))
if FILTER_DATA:
    # Simple filtering based on RGB value thresholding
    idx = [i for i in idx if not(parted0[i].mean(axis=(0,1))[2] > 160 and parted0[i].mean(axis=(0,1))[1] < 150 and parted0[i].mean(axis=(0,1))[0] < 100 or parted0[i].mean(axis=(0,1))[2] > 200)]
    idx = [i for i in idx if not(parted0[i].var(axis=(0,1)).mean() < 100)]
val_idxs = np.random.choice(idx, val_size,replace=False)
print("Saving {}% of the complete image. Number of images: {}".format(int(100*(len(idx) / len(parted0))), len(idx)))
iters = 0
if USE_ROADMAP_AS_TARGET:
    print("Using ROADMAP as target image")
else:
    print("Using ATLUS as target image")
for i in idx:
    image_cat = "train"
    if i in val_idxs:
        image_cat = "val"
    savedir = os.path.join(save_path, image_cat)

    if USE_ROADMAP_AS_TARGET:
        target_im = Image.fromarray(parted0[i])
    else:
        target_im = Image.fromarray(parted1[i])
    input_image = Image.fromarray(parted2[i])

    save_aligned(input_image, target_im, os.path.join(savedir, str(iters) + '_0.png'))
    if AUGMENT_DATA:
        save_aligned(
            input_image.transpose(Image.FLIP_LEFT_RIGHT),
            target_im.transpose(Image.FLIP_LEFT_RIGHT),
            os.path.join(savedir, "{}_1.png".format(iters))
        )
        save_aligned(
            input_image.transpose(Image.FLIP_TOP_BOTTOM),
            target_im.transpose(Image.FLIP_TOP_BOTTOM),
            os.path.join(savedir, "{}_2.png".format(iters))
        )
        for rotate in [90, 180, 270]:
            save_aligned(
                input_image.rotate(rotate),
                target_im.rotate(rotate),
                os.path.join(savedir, "{}_{}_0.png".format(iters, rotate))
            )
            save_aligned(
                input_image.rotate(rotate).transpose(Image.FLIP_LEFT_RIGHT),
                target_im.rotate(rotate).transpose(Image.FLIP_LEFT_RIGHT),
                os.path.join(savedir, "{}_{}_1.png".format(iters, rotate))
            )
            save_aligned(
                input_image.rotate(rotate).transpose(Image.FLIP_TOP_BOTTOM),
                target_im.rotate(rotate).transpose(Image.FLIP_TOP_BOTTOM),
                os.path.join(savedir, "{}_{}_2.png".format(iters, rotate))
            )
    iters += 1


print("Generating chopped up image..")
im = np.zeros((8192,8192, 3))
j = 0
for row in range(8192//M):
    for col in range(8192//M):
        if j in idx:
            ims = parted1[j]
            im[row*M:M*(row+1), col*N:N*(col+1), :] = parted0[j]
        j+= 1

name = "GTA_choppedup.png"
path = os.path.join(original_image_path, name)

plt.imsave(path,im/255 )
print("Chopped up image saved in:", path)              


