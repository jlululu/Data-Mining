# -*- coding: utf-8 -*-
import imageio
import skimage.color as color
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage.feature as feature
import skimage.filters as filters
import skimage.transform as transform
from skimage.util import random_noise
from skimage.segmentation import slic, mark_boundaries

# 1.
avenger = imageio.imread('data/avengers_imdb.jpg')
print('The size is: ',avenger.size)
# convert to grayscale format
avenger_gray = color.rgb2gray(avenger)
# convert to binary format
thresh = filters.threshold_otsu(avenger_gray)
# change the pixel values larger than threshold to 1, otherwise, change to 0
avenger_binary = avenger_gray > thresh
# display the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(avenger, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original image', fontsize=20)

ax2.imshow(avenger_gray, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('grayscale image', fontsize=20)

ax3.imshow(avenger_binary, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('binary image', fontsize=20)

fig.tight_layout()
plt.show()



# 2.
bush = imageio.imread('data/bush_house_wikipedia.jpg')
# add Gaussian random noise
bush_noise = random_noise(bush, mode='gaussian',var=0.1)
# filter with a Gaussian mask
bush_gaussian = ndi.gaussian_filter(bush_noise, 1)
# filter with a uniform smoothing mask
bush_uniform = ndi.uniform_filter(bush_noise,size=9)
# display the results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 4), sharex=True, sharey=True)

ax1.imshow(bush, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=10)

ax2.imshow(bush_noise, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('noisy', fontsize=10)

ax3.imshow(bush_gaussian, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Gaussian mask', fontsize=10)

ax4.imshow(bush_uniform, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('uniform mask', fontsize=10)

fig.tight_layout()
plt.show()



# 3.
forestry = imageio.imread('data/forestry_commission_gov_uk.jpg')
# k-means segmentation
segments = slic(forestry, n_segments=5, compactness=20, sigma=1, start_label=1)
# return segmented image
out = mark_boundaries(forestry,segments,outline_color=1)
# display the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(forestry, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original image', fontsize=20)

ax2.imshow(out)
ax2.axis('off')
ax2.set_title('segmented image', fontsize=20)

ax3.imshow(segments)
ax3.axis('off')
ax3.set_title('segments', fontsize=20)

fig.tight_layout()
plt.show()



# 4.
rolland = imageio.imread('data/rolland_garros_tv5monde.jpg')
# for better performance, use a Gaussian mask to smooth the image
rolland = ndi.gaussian_filter(rolland, 1)
# convert to grayscale format
rolland_gray = color.rgb2gray(rolland)
# Canny edge detection
edges = feature.canny(rolland_gray)
# Hough transform
lines = transform.probabilistic_hough_line(edges,threshold=3,line_length=40,line_gap=3)
# display the results
plt.figure()
plt.title('edge detection')
plt.imshow(edges)
for line in lines:
    p0, p1 = line
    plt.plot((p0[0],p1[0]),(p0[1],p1[1]))
plt.show()
