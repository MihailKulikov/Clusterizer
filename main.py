from LoadDataset import get_dataset
from GetModel import get_model
from efficientnet_pytorch import EfficientNet
import torch
from ImageHandlers import print_image_from_ndarray
import numpy as np
from sklearn.cluster import KMeans

all_data = get_dataset('./Orbits')
model = get_model('efficientnet-b0')
print(all_data[0][0].shape)
with torch.no_grad():
    patients_features = torch.stack(list(map(lambda patient_data: torch.stack(list(map(lambda image: model(image),
                                                                                       patient_data))),
                                             all_data))).reshape(2, 7680).numpy()
kmeans = KMeans(n_clusters=3).fit(patients_features)
print(kmeans.labels_)

