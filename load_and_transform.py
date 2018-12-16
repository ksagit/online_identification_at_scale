import torchvision
import torch
import numpy as np
import os
import PIL.Image

DATASET_PATH = '/home/kylesargent/test/'

def transform_image(pil_image):
    img = torchvision.transforms.Resize(256)(pil_image)
    img = torchvision.transforms.RandomCrop(224)(img)
    img = torchvision.transforms.RandomGrayscale(p=0.2)(img)

    img = np.array(img, dtype=np.uint8)
    img = torch.from_numpy(img)
    
    img = img.type('torch.FloatTensor')
    img = img.transpose(1,2)
    img = img.transpose(0,1)
    return img

def get_image_list():
    images = []
    for person in os.listdir(DATASET_PATH):
        for image in os.listdir(os.path.join(DATASET_PATH, person)):
            images += [[os.path.join(DATASET_PATH, person, image), person]]
    return np.array(images)

def load_image_batch(image_filenames):
    batch_images = map(lambda filename: PIL.Image.open(filename), image_filenames)
    batch_images = map(transform_image, batch_images)
    batch_images = map(lambda x : x.unsqueeze(0), batch_images)
    batch_images = torch.cat(batch_images, 0)
    batch_images = batch_images.cuda()
    return batch_images