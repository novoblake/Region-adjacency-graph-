import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation
from skimage import graph
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from keras.layers import Input, Dense
from keras.models import Model

img = cv2.imread(r'C:\Users\Ishan\Projects\im0l.png')
labels = segmentation.slic(img, compactness=20, n_segments=100, start_label=1)
g = graph.rag_mean_color(img, labels)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].set_title('RAG drawn with default settings')
lc = graph.show_rag(labels, g, img, ax=ax[0])
# specify the fraction of the plot area that will be used to draw the colorbar
fig.colorbar(lc, fraction=0.03, ax=ax[0])

ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
lc = graph.show_rag(labels, g, img,
                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])
fig.colorbar(lc, fraction=0.03, ax=ax[1])

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
print(g.nodes)
print(g.edges)
print(g.degree)
print(g.adj)
print(g.__class__)

