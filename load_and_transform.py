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

def pairwise_distances(x,y):
    x_norms = torch.sum(x*x, dim=1).unsqueeze(1)
    y_norms = torch.sum(y*y, dim=1).unsqueeze(0)    
    return x_norms - 2*torch.mm(x, y.transpose(0, 1)) + y_norms

def get_image_list(n_people=len(os.listdir(DATASET_PATH)), n_images_per_person=1000):
    all_images = []
    for person in np.random.choice(os.listdir(DATASET_PATH), n_people, replace=False):
        image_files = os.listdir(os.path.join(DATASET_PATH, person))
        if len(image_files) > n_images_per_person:
            person_images = np.random.choice(image_files, n_images_per_person, replace=False)
        else:
            person_images = image_files
            
        for image in person_images:
            all_images += [[os.path.join(DATASET_PATH, person, image), person]]
    return np.array(all_images)

def load_image_batch(image_filenames):
    batch_images = map(lambda filename: PIL.Image.open(filename), image_filenames)
    batch_images = map(transform_image, batch_images)
    batch_images = map(lambda x : x.unsqueeze(0), batch_images)
    batch_images = torch.cat(batch_images, 0)
    batch_images = batch_images.cuda()
    return batch_images

