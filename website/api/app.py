"""
@ version: app.py v1.0.0
@ author: Gaole Dai (20124917)
@ email: scygd1@nottingham.edu.cn
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import re
import cv2
from numpy.testing import assert_almost_equal
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil

import torch
from torch.cuda import device
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

app = Flask(__name__)
# Turn on the debug mode
app.config['DEBUG'] = True
CORS(app)

search_id = ""
with open('../models/similar_names_400.pkl', 'rb') as f:
    similar_names_400 = pickle.load(f)
with open('../models/similar_values_400.pkl', 'rb') as f:
    similar_values_400 = pickle.load(f)

with open('../models/similar_names_100.pkl', 'rb') as f:
    similar_names_100 = pickle.load(f)
with open('../models/similar_values_100.pkl', 'rb') as f:
    similar_values_100 = pickle.load(f)

with open('../models/vlad_descriptors.pkl', 'rb') as f:
    vlad_descriptors = pickle.load(f)
with open('../models/kmeans_clusters.pkl', 'rb') as f:
    kmeans_clusters = pickle.load(f)

with open('../models/cbir_similar_vectors_100.pkl', 'rb') as f:
    similar_cluster_encoder = pickle.load(f)

DEPTH = 5


@app.route("/show_all_image/", methods=['GET'])
def show_image():
    mypath = "../../oxbuild_images_100"
    filenames = next(os.walk(mypath), (None, None, []))[2]
    return jsonify(filenames)


@app.route("/search_images/<string:keywords>", methods=['GET'])
def keyword_image(keywords):
    mypath = "../../oxbuild_images_100"
    valid = re.sub(r"[^A-Za-z]+", '', keywords)
    filenames = [f for f in os.listdir(mypath) if re.match(rf'^.*{valid}.*\.jpg$', f, re.IGNORECASE)]
    return jsonify(filenames)


@app.route("/image_id/<string:image_id>", methods=['POST'])
def query_image(image_id):
    global search_id
    search_id = image_id
    return jsonify({"status": "success"})


@app.route("/query_image_id/", methods=['GET'])
def query_image_id():
    global search_id
    return jsonify(search_id)


@app.route("/query_similar_image_cnn/", methods=['GET'])
def query_similar_img_cnn():
    global search_id
    global similar_names_400
    global similar_values_400
    if search_id in set(similar_names_400.index):
        imgs = list(similar_names_400.loc[search_id, :])
        vals = list(similar_values_400.loc[search_id, :])
        if search_id in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(search_id)
            vals.remove(max(vals))
            return jsonify([{'imgs': imgs, 'vals': vals}])


@app.route("/query_similar_image_cnn100/", methods=['GET'])
def query_similar_img_cnn100():
    global search_id
    global similar_values_100
    global similar_names_100
    if search_id in set(similar_names_100.index):
        imgs = list(similar_names_100.loc[search_id, :])
        vals = list(similar_values_100.loc[search_id, :])
        if search_id in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(search_id)
            vals.remove(max(vals))
            return jsonify([{'imgs': imgs, 'vals': vals}])


@app.route("/query_similar_image_encoder/", methods=['GET'])
def query_similar_img_encoder():
    global search_id
    global similar_cluster_encoder

    values = similar_cluster_encoder.get(search_id)
    return jsonify(values)


@app.route("/query_similar_image_sift/", methods=['GET'])
def query_similar_img_sift():
    global search_id
    global kmeans_clusters
    global vlad_descriptors

    list_res = []
    # compute SIFT descriptor for query
    img = cv2.imread(os.path.join('../../oxbuild_images_100', search_id))
    descriptor = describe_SIFT(img)

    # compute VLAD descriptor for query
    v = compute_vlad_descriptor(descriptor, kmeans_clusters)

    # Get distances between query VLAD and dataset VLADs descriptors
    for i in range(len(vlad_descriptors)):
        temp_vec = vlad_descriptors[i]['feature_vector']
        dist = np.linalg.norm(temp_vec - v)
        list_res.append({'i': i,
                         'dist': dist,
                         'image_name': vlad_descriptors[i]['image_name']})
    res_ = sorted(list_res, key=lambda x: x['dist'])

    return jsonify(res_[:DEPTH])


@app.route("/input_image_search/<string:filename>", methods=['GET'])
def input_img_search(filename):
    shutil.move("../" + filename, '../../oxbuild_images_100')
    os.remove('../../oxbuild_images_100'+filename)
    # Resize images
    input_dim = (224, 224)
    input_path = '../../oxbuild_images_100'
    output_path = '../../oxbuild_images_100_new'

    os.makedirs(output_path, exist_ok=True)

    img_transformation = transforms.Compose([transforms.Resize(input_dim)])

    for image_name in os.listdir(input_path):
        img = Image.open(os.path.join(input_path, image_name))
        new_img = img_transformation(img)

        new_img.save(os.path.join(output_path, image_name))

        new_img.close()
        img.close()

    # generate vectors for all the images in the set
    img_vectors = img2VectorResnet18()

    all_vectors = {}
    print("Converting images to feature vectors:")
    for image in tqdm(os.listdir('../../oxbuild_images_100_new')):
        img = Image.open(os.path.join('../../oxbuild_images_100_new', image))
        vec = img_vectors.get_feature_vector(img)
        all_vectors[image] = vec
        img.close()

    similarity_matrix = get_similarity_matrix(all_vectors)

    k = 5  # the number of top similar images to be stored

    similar_names = pd.DataFrame(index=similarity_matrix.index, columns=range(k))
    similar_values = pd.DataFrame(index=similarity_matrix.index, columns=range(k))

    for j in tqdm(range(similarity_matrix.shape[0])):
        k_similar = similarity_matrix.iloc[j, :].sort_values(ascending=False).head(k)
        similar_names.iloc[j, :] = list(k_similar.index)
        similar_values.iloc[j, :] = k_similar.values

    if filename in set(similar_names.index):
        imgs = list(similar_names.loc[filename, :])
        vals = list(similar_values.loc[filename, :])
        if filename in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(filename)
            vals.remove(max(vals))
            return jsonify([{'imgs': imgs, 'vals': vals}])


class img2VectorResnet18:
    def __init__(self):
        if device == 'cuda':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.features_number = 512
        self.model, self.featureLayer = self.get_feature_layer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

    def get_feature_vector(self, img):
        image = self.toTensor(img).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.features_number, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def get_feature_layer(self):
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 11)

        if device == 'cuda':
            model_ft.cuda()

        model_ft.load_state_dict(
            torch.load('../../models/fine_tuned_best_model_400.pth', map_location=torch.device(device)))
        model_ft.eval()
        layer = model_ft._modules.get('avgpool')
        self.layer_output_size = 512

        return model_ft, layer


def get_similarity_matrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / (
            (np.linalg.norm(v, axis=0).reshape(-1, 1)) * ((np.linalg.norm(v, axis=0).reshape(-1, 1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns=keys, index=keys)

    return matrix


def describe_SIFT(image):
    """
  :param image: image path
  Calculate SIFT descriptor with nfeatures key-points
  """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # keypoints, descriptors = sift.detect(image, None)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


def compute_vlad_descriptor(descriptors, kmeans_clusters):
    """
  :param descriptor: SIFT descriptor of image
  :param kmeans_clusters: Object of Kmeans (sklearn)
  First we need to predict clusters fot key-points of image (row in
  input descriptor). Then for each cluster we get descriptors, which belong to it,
  and calculate sum of residuals between descriptor and centroid (cluster center)
  """
    # Get SIFT dimension (default: 128)
    sift_dim = descriptors.shape[1]

    # Predict clusters for each key-point of image
    labels_pred = kmeans_clusters.predict(descriptors)

    # Get centers fot each cluster and number of clusters
    centers_cluster = kmeans_clusters.cluster_centers_
    numb_cluster = kmeans_clusters.n_clusters
    vlad_descriptors = np.zeros([numb_cluster, sift_dim])

    # Compute the sum of residuals (for belonging x for cluster) for each cluster
    for i in range(numb_cluster):
        if np.sum(labels_pred == i) > 0:
            # Get descritors which belongs to cluster and compute residuals between x and centroids
            x_belongs_cluster = descriptors[labels_pred == i, :]
            vlad_descriptors[i] = np.sum(x_belongs_cluster - centers_cluster[i], axis=0)

    # Create vector from matrix
    vlad_descriptors = vlad_descriptors.flatten()

    # Power and L2 normalization
    vlad_descriptors = np.sign(vlad_descriptors) * (np.abs(vlad_descriptors) ** (0.5))
    vlad_descriptors = vlad_descriptors / np.sqrt(vlad_descriptors @ vlad_descriptors)
    return vlad_descriptors


if __name__ == '__main__':
    app.run(threaded=True)
