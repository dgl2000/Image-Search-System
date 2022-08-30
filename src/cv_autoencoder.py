# -*- coding: utf-8 -*-
"""cv_autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hRuXG5f6uK5THEcVDhcz2OzpDvmXoMsW

# COMP3065 Computer Vision Coursework
@author: Gaole Dai (20124917)

@email: scygd1@nottingham.edu.cn

@file: image search with encoder and decoder (CNN) for feature extraction

@cite: code partially from https://github.com/oke-aditya/image_similarity

## Mount Drive
"""


"""## Prerequisites"""

import os
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from torch.utils.data import Dataset

from torchvision import transforms
import torch.optim as optim
from torchvision.io import read_image


if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

"""## Images Resize"""

inputDim = (224, 224)
input_path = '/content/drive/MyDrive/y4_cv/oxbuild_images_100'
output_path = '/content/drive/MyDrive/y4_cv/oxbuild_images_100_new'

os.makedirs(output_path, exist_ok = True)

img_transformation = transforms.Compose([transforms.Resize(inputDim)])

for image_name in os.listdir(input_path):
    img = Image.open(os.path.join(input_path, image_name))
    new_img = img_transformation(img)

    # # copy the rotation information metadata from original image and save, else your transformed images may be rotated
    # exif = img.info['exif']
    new_img.save(os.path.join(output_path, image_name))
    
    new_img.close()
    img.close()

"""## Dataset Preparation and DataLoader"""

labels_map = {
    0: "all-souls",
    1: "ashmolean",
    2: "balliol",
    3: "bodleian",
    4: "christ church",
    5: "cornmarket",
    6: "hertford",
    7: "keble",
    8: "magdalen",
    9: "pitt rivers",
    10: "radcliffe camera"
}

all_imgs = []

class OxfordBuilding(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_dir = '/content/drive/MyDrive/y4_cv/oxbuild_images_100_new'
        self.transform = transform
        self.target_transform = target_transform
        self.all_imgs = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.all_imgs[idx])
        image = read_image(img_path)
        if self.transform:
          image = self.transform(image)
        return image, image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.4814, 0.4865, 0.4683], [0.2736, 0.2776, 0.3152])
])

oxford5k_dataset = OxfordBuilding(transform=transform)
test_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and test splits:
dataset_size = len(oxford5k_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
print(dataset_size)

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(oxford5k_dataset, batch_size=32, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(oxford5k_dataset, batch_size=16, sampler=test_sampler)
full_loader = torch.utils.data.DataLoader(oxford5k_dataset, batch_size=dataset_size, shuffle=False)

# """
# # Calculate the std and mean
# def batch_mean_and_sd(loader):
#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)

#     for images, _ in loader:
#         b, c, h, w = images.shape
#         nb_pixels = b * h * w
#         sum_ = torch.sum(images, dim=[0, 2, 3])
#         sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
#         fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#         snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
#         cnt += nb_pixels

#     mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
#     return mean,std
  
# mean, std = batch_mean_and_sd(train_loader)
# print("mean and std: \n", mean, std)
# """

"""## Autoencoder Model Initial"""

class Encoder(nn.Module):
  def __init__(self):
      super().__init__()
      # self.img_size = img_size
      self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
      self.relu1 = nn.ReLU(inplace=True)
      self.maxpool1 = nn.MaxPool2d((2, 2))

      self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
      self.relu2 = nn.ReLU(inplace=True)
      self.maxpool2 = nn.MaxPool2d((2, 2))

      self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
      self.relu3 = nn.ReLU(inplace=True)
      self.maxpool3 = nn.MaxPool2d((2, 2))

      self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
      self.relu4 = nn.ReLU(inplace=True)
      self.maxpool4 = nn.MaxPool2d((2, 2))

  def forward(self, x):
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.maxpool1(x)

      x = self.conv2(x)
      x = self.relu2(x)
      x = self.maxpool2(x)

      x = self.conv3(x)
      x = self.relu3(x)
      x = self.maxpool3(x)

      x = self.conv4(x)
      x = self.relu4(x)
      x = self.maxpool4(x)

      return x


class Decoder(nn.Module):
  def __init__(self):
      super().__init__()
      self.deconv1 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
      self.relu1 = nn.ReLU(inplace=True)

      self.deconv2 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
      self.relu2 = nn.ReLU(inplace=True)

      self.deconv3 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
      self.relu3 = nn.ReLU(inplace=True)

      self.deconv4 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
      self.relu4 = nn.ReLU(inplace=True)

  def forward(self, x):
      x = self.deconv1(x)
      x = self.relu1(x)

      x = self.deconv2(x)
      x = self.relu2(x)

      x = self.deconv3(x)
      x = self.relu3(x)
      
      x = self.deconv4(x)
      x = self.relu4(x)
      return x

"""## Training"""

def train_step(encoder, decoder, loss_fn, optimizer, device):
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = loss_fn(dec_output, target_img)
        loss.backward()

        optimizer.step()

    return loss.item()


def test_step(encoder, decoder, loss_fn, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
      for batch_idx, (train_img, target_img) in enumerate(test_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = loss_fn(dec_output, target_img)

    return loss.item()

loss_fn = nn.MSELoss()

encoder = Encoder()
decoder = Decoder()

encoder.to(device)
decoder.to(device)

autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.AdamW(autoencoder_params, lr=0.0001)

max_loss = 9999

print("------------ Training started ------------")

for epoch in tqdm(range(60)):
    train_loss = train_step(
        encoder, decoder, loss_fn, optimizer, device=device
    )
    print(f"Epochs = {epoch}, Training Loss : {train_loss}")
    test_loss = test_step(
        encoder, decoder, loss_fn, device=device
    )

    # Simple Best Model saving
    if test_loss < max_loss:
        print("Validation Loss decreased, saving new best model")
        torch.save(encoder.state_dict(), '/content/drive/MyDrive/y4_cv/encoder_100_2.pth')
        torch.save(decoder.state_dict(), '/content/drive/MyDrive/y4_cv/decoder_100_2.pth')

    print(f"Epochs = {epoch}, Validation Loss : {test_loss}")

print("Training Done")

"""## Feature Extraction"""

model_encoder = Encoder()
model_encoder.load_state_dict(
    torch.load('/content/drive/MyDrive/y4_cv/encoder_100_2.pth'))
model_encoder.to(device)

embedding_dim = (1, 128, 14, 14)
def create_embedding(encoder, full_loader, embedding_dim, device):
  encoder.eval()
  embedding = torch.randn(embedding_dim).to(device)

  with torch.no_grad():
    for batch_idx, (train_img, target_img) in enumerate(full_loader):
        train_img = train_img.to(device)
        enc_output = encoder(train_img).to(device)
        embedding = torch.cat((embedding, enc_output), 0)
  return embedding

embedding = create_embedding(model_encoder, full_loader, embedding_dim, device)
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]

flattened_embedding = numpy_embedding.reshape((num_images, -1))

"""## Compute Similar Images"""

def compute_similar_images(model, image_tensor, num_images, embedding, device):
  image_tensor = image_tensor.to(device)

  with torch.no_grad():
    image_embedding = model(image_tensor).cpu().detach().numpy()

  flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

  knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")

  knn.fit(embedding)

  _, indices = knn.kneighbors(flattened_embedding)
  indices_list = indices.tolist()
  # print(indices_list)
  return indices_list

img_names = oxford5k_dataset.all_imgs

all_vectors = {}
for image in tqdm(os.listdir('/content/drive/MyDrive/y4_cv/oxbuild_images_100_new')):
    img = Image.open(os.path.join('/content/drive/MyDrive/y4_cv/oxbuild_images_100_new', image))
    image_tensor = transforms.ToTensor()(img)
    image_tensor = image_tensor.unsqueeze(0)
    indices_list = compute_similar_images(
        model_encoder, image_tensor, 4, embedding=flattened_embedding, device=device
    )
    all_vectors[image] = [img_names[(indices_list[0][0]-1)], img_names[(indices_list[0][1]-1)], 
                img_names[(indices_list[0][2]-1)], img_names[(indices_list[0][3]-1)]]
    img.close()
print(all_vectors)
with open('/content/drive/MyDrive/y4_cv/cbir_similar_vectors_100_2.pkl', 'wb') as f:
  pickle.dump(all_vectors, f)

"""## Option 2 for Image Cluster"""

def create_embedding(image, encoder, device):
    encoder.eval()
    with torch.no_grad():
      toTensor = transforms.ToTensor()
      image = toTensor(image).unsqueeze(0).to(device)
      # embedding = torch.zeros(1, self.features_number, 1, 1)

      def copyData(m, i, o): embedding.copy_(o.data)
      embedding = encoder(image).to(device)
      # embedding = torch.cat((embedding, enc_output), 0)
      # print(embedding.shape)
    return embedding.cpu().detach().numpy()[0, :, 0, 0]
    

all_vectors = {}
print("Converting images to feature vectors:")
for image in tqdm(os.listdir('/content/drive/MyDrive/y4_cv/oxbuild_images_100_new')):
    img = Image.open(os.path.join('/content/drive/MyDrive/y4_cv/oxbuild_images_100_new', image))
    vec = create_embedding(img, encoder, device)
    all_vectors[image] = vec
    img.close()

def get_similarity_matrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    return matrix

similarity_matrix = get_similarity_matrix(all_vectors)

k = 5 # the number of top similar images to be stored

similar_names = pd.DataFrame(index = similarity_matrix.index, columns = range(k))
similar_values = pd.DataFrame(index = similarity_matrix.index, columns = range(k))

for j in tqdm(range(similarity_matrix.shape[0])):
  k_similar = similarity_matrix.iloc[j, :].sort_values(ascending = False).head(k)
  similar_names.iloc[j, :] = list(k_similar.index)
  similar_values.iloc[j, :] = k_similar.values
    
similar_names.to_pickle('/content/drive/MyDrive/y4_cv/similar_names_model4.pkl')
similar_values.to_pickle('/content/drive/MyDrive/y4_cv/similar_values_model4.pkl')

inputImages = ["all_souls_000013.jpg", "ashmolean_000058.jpg", "hertford_000015.jpg", "christ_church_000179.jpg"]

numCol = 5
numRow = 1

def setAxes(ax, image, query = False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize = 12)
    else:
        ax.set_xlabel("Similarity value {1:1.3f}\n{0}".format(image, value), fontsize = 12)
    ax.set_xticks([])
    ax.set_yticks([])
    
def get_similar_images(image, simNames, simVals):
    if image in set(simNames.index):
      imgs = list(simNames.loc[image, :])
      vals = list(simVals.loc[image, :])
      if image in imgs:
        assert_almost_equal(max(vals), 1, decimal = 5)
        imgs.remove(image)
        vals.remove(max(vals))
      return imgs, vals
    else:
      print("'{}' Unknown image".format(image))
        
def plot_similar_images(image, simiar_names, similar_values):
    simImages, simValues = get_similar_images(image, similar_names, similar_values)
    fig = plt.figure(figsize=(10, 20))

    for j in range(0, numCol*numRow):
      ax = []
      if j == 0:
        img = Image.open(os.path.join('/content/drive/MyDrive/y4_cv/oxbuild_images_100', image))
        ax = fig.add_subplot(numRow, numCol, 1)
        setAxes(ax, image, query = True)
      else:
        img = Image.open(os.path.join('/content/drive/MyDrive/y4_cv/oxbuild_images_100', simImages[j-1]))
        ax.append(fig.add_subplot(numRow, numCol, j+1))
        img = img.convert('RGB')
      plt.imshow(img)
      img.close()
    plt.show()   

for image in inputImages:
    plot_similar_images(image, similar_names, similar_values)