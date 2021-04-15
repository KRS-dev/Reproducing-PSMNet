#%%

import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from IO import *



disp = mpimg.imread(r'output\test.tiff')
left = mpimg.imread(r'E:\TU-delft\CS4240 - Deep Learning\PSMNet\dataset\sceneflow example\Sampler\Driving\RGB_cleanpass\left\0400.png')
right = mpimg.imread(r'E:\TU-delft\CS4240 - Deep Learning\PSMNet\dataset\sceneflow example\Sampler\Driving\RGB_cleanpass\right\0400.png')

fig, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(20,5))
axes[0].imshow(right)
axes[1].imshow(left)
ax = axes[2].imshow(disp)
fig.colorbar(ax)

# %%

pfm_data = readPFM(r'E:\TU-delft\CS4240 - Deep Learning\PSMNet\dataset\sceneflow example\Sampler\Driving\disparity\0400.pfm')[0]

fig, axes = plt.subplots(1,2, sharex=True, sharey=True, figsize=(15,10))
ax1 = axes[0].imshow(pfm_data, vmin=0, vmax=270)
ax2 = axes[1].imshow(disp*1.17, vmin=0, vmax=270)
fig.colorbar(ax1, ax=axes[0], orientation='horizontal')
fig.colorbar(ax1, ax=axes[1], orientation='horizontal')

axes[0].set_xlabel('Ground Truth')
axes[1].set_xlabel('Predicted')
plt.show()

# %%
