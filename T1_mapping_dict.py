import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from skimage.filters import gaussian
from einops import rearrange
import os

data_path = "T1 IR"
TI_list = []
img_list = []
for data in os.listdir(data_path):
    if 'tse' in data:
        TI = int(data.split('_')[-1])
        img = dcmread(os.path.join(data_path, data, '00001.dcm')).pixel_array
        TI_list.append(TI)
        img_list.append(img)
    
TI_list = np.asarray(TI_list)
img_list = np.asarray(img_list)
TI_list_idx = np.argsort(TI_list)
TI_list = TI_list[TI_list_idx]
img_list = img_list[TI_list_idx]

# Filter the image 
img_preprocess = gaussian(img_list, sigma=0.8, channel_axis=0)
# img_preprocess = img_list.copy().astype(np.float64)
img_preprocess/=img_preprocess.max()

plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(img_list[i], cmap='gray')
    plt.clim(img_list.min(), img_list.max())
    plt.colorbar()

plt.figure()
for i in range(len(img_list)):
    plt.subplot(2,5,i+1)
    plt.imshow(img_preprocess[i], cmap='gray')
    plt.clim(img_preprocess.min(), img_preprocess.max())
    plt.colorbar()
plt.show()

t, h, w = img_preprocess.shape
img_preprocess_flatten = rearrange(img_preprocess, "t h w -> t (h w)")
print(img_preprocess_flatten.shape)

# Generate Dict
T1_keys = np.arange(50, 2301, 0.1)

T1_dict = []
for TI in TI_list:
    T1_dict.append(np.abs((1-2*np.exp(-TI/T1_keys))))

T1_dict = np.asarray(T1_dict).T

print(T1_dict.shape)

U,S,VH = np.linalg.svd(T1_dict,full_matrices=False)
# plt.plot(np.cumsum(S)/np.sum(S))
# plt.show()

# for i,k_feature in enumerate(range(3,11)):
k_feature = 8

VH_k = VH[:k_feature]
U_k = U[:,:k_feature]

T1_dict_k = T1_dict@VH_k.T
img_preprocess_flatten_k = img_preprocess_flatten.T@VH_k.T
print(T1_dict_k.shape,img_preprocess_flatten_k.shape)

print(img_preprocess_flatten_k.sum(axis=1).shape, T1_dict_k.sum(axis=1).shape)
print(np.outer(img_preprocess_flatten_k.sum(axis=1),T1_dict_k.sum(axis=1)).shape)

map = np.abs((img_preprocess_flatten_k @ T1_dict_k.T*k_feature-np.outer(img_preprocess_flatten_k.sum(axis=1),T1_dict_k.sum(axis=1)))/(np.sqrt(np.outer(k_feature*(img_preprocess_flatten_k**2).sum(axis=1)-(img_preprocess_flatten_k.sum(axis=1))**2,k_feature*(T1_dict_k**2).sum(axis=1)-(T1_dict_k.sum(axis=1))**2))))
print(map.shape)
map_max = np.argmax(map, axis=1)

T1_map = T1_keys[map_max]
T1_map_img = rearrange(T1_map, "(h w) -> h w", h=h, w=w)
print(T1_map_img.shape)

plt.imshow(T1_map_img)
plt.clim(0,2250)
plt.colorbar()
plt.show()