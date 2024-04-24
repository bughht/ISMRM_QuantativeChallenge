import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from skimage.filters import gaussian
from einops import rearrange
import os

data_path = "T2star"
TE_list = []
img_list = []
for data in os.listdir(data_path):
    if 'TE' in data:
        TE = int(data.split('_')[-1][2:-2])
        img = dcmread(os.path.join(data_path, data, '00001.dcm')).pixel_array
        TE_list.append(TE)
        img_list.append(img)
    
TE_list = np.asarray(TE_list)
img_list = np.asarray(img_list)
TE_list_idx = np.argsort(TE_list)
TE_list = TE_list[TE_list_idx]
img_list = img_list[TE_list_idx]

print(TE_list, img_list.shape)

print(TE_list)

# Filter the image 
img_preprocess = gaussian(img_list, sigma=0.8, channel_axis=0)
# img_preprocess = img_list.copy().astype(np.float64)
img_preprocess/=img_preprocess.max()

# plt.figure()
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(img_list[i], cmap='gray')
#     plt.clim(img_list.min(), img_list.max())
#     plt.colorbar()

# plt.figure()
# for i in range(len(img_list)):
#     plt.subplot(3,4,i+1)
#     plt.imshow(img_preprocess[i], cmap='gray')
#     plt.clim(img_preprocess.min(), img_preprocess.max())
#     plt.colorbar()
# plt.show()

t, h, w = img_preprocess.shape
img_preprocess_flatten = rearrange(img_preprocess, "t h w -> t (h w)")
print(img_preprocess_flatten.shape)

# Generate Dict
T2star_keys = np.arange(0.1, 500, 0.05)

T2star_dict = []
for TE in TE_list:
    T2star_dict.append(np.abs((np.exp(-TE/T2star_keys))))

T2star_dict = np.asarray(T2star_dict).T

print(T2star_dict.shape)

U,S,VH = np.linalg.svd(T2star_dict,full_matrices=False)
# plt.plot(np.cumsum(S)/np.sum(S))
# plt.show()

# for i,k_feature in enumerate(range(3,11)):
k_feature = 4

VH_k = VH[:k_feature]
U_k = U[:,:k_feature]

T2star_dict_k = T2star_dict@VH_k.T
img_preprocess_flatten_k = img_preprocess_flatten.T@VH_k.T
print(T2star_dict_k.shape,img_preprocess_flatten_k.shape)

print(img_preprocess_flatten_k.sum(axis=1).shape, T2star_dict_k.sum(axis=1).shape)
print(np.outer(img_preprocess_flatten_k.sum(axis=1),T2star_dict_k.sum(axis=1)).shape)

map = np.abs((img_preprocess_flatten_k @ T2star_dict_k.T*k_feature-np.outer(img_preprocess_flatten_k.sum(axis=1),T2star_dict_k.sum(axis=1)))/(np.sqrt(np.outer(k_feature*(img_preprocess_flatten_k**2).sum(axis=1)-(img_preprocess_flatten_k.sum(axis=1))**2,k_feature*(T2star_dict_k**2).sum(axis=1)-(T2star_dict_k.sum(axis=1))**2))))
print(map.shape)
map_max = np.argmax(map, axis=1)

T2star_map = T2star_keys[map_max]
T2star_map_img = rearrange(T2star_map, "(h w) -> h w", h=h, w=w)
print(T2star_map_img.shape)

plt.imshow(T2star_map_img)
plt.clim(0,500)
plt.colorbar()
plt.show()