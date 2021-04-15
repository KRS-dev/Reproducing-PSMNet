# Introduction to disparity maps

PSMNet is a Stereo Matching Network published by .... in 2018. The goal
of stereo matching is to find the disparity map of a left and right
image. Disparity is in direct relation to depth through the focal
distance and baseline of the two lenses used. The depth is equal to
focallength\*baseline/disparity so a high disparity means that the
object is close to the camera’s and low disparity means it is far away.

<figure>
<img src="images/baseline.png" id="fig:my_label" alt="Disparity to depth" /><figcaption aria-hidden="true">Disparity to depth</figcaption>
</figure>

This equation does assume that the images are rectified, meaning the
camera’s are exactly parallel to each other, not rotated inwards or
outwards. Normal stereo camera setups do have this parallelism, but some
camera arrays need a prepossessing step rectifying the images to a
common plane.

<figure>
<img src="images/Lecture_1027_stereo_01.jpg" id="fig:my_label" alt="Image rectification" /><figcaption aria-hidden="true">Image rectification</figcaption>
</figure>

An example of a disparity map is given below. Deep learning networks
have been used for a while to learn disparity maps from stereo images
and today we are going to discuss reproducing PSMNet, a high performing
Pyramid network on both Sceneflow and KITTI 2012/2015 stereo data-sets.

<figure>
<img src="images/example_disparity.png" id="fig:my_label" alt="Disparity map created by PSMNet. Images are from the sceneflow dataset." /><figcaption aria-hidden="true">Disparity map created by PSMNet. Images are from the sceneflow dataset.</figcaption>
</figure>

# Pyramid network

<figure>
<img src="images/architecture.PNG" id="fig:architecture" alt="Architecture of PSMNet" /><figcaption aria-hidden="true">Architecture of PSMNet</figcaption>
</figure>

PSMNet consists of two main parts, the pyramid structured feature
extractor and a 3D CNN. The pyramid feature extractor tries to find a
different sized features through different pooling sizes (8x8, 16x16,
32x32, 64x64). These pooled features are led through convolutions where
after they are up-sampled to the same HxW dimensions again. Both image’s
features are combined into a 4D cost volume for each disparity level
(HeightxWidthxFeaturexDisparity). The 3D CNN consists out of a multiple
stackhourglass type 3D convolutions with residual connections. The final
disparity is calculated using regression with the following formula
[\[eq1:regression\]][1], where *D*<sub>*m**a**x*</sub> is the maximum
disparity the model can predict, *c*<sub>*d*</sub> the predicted cost
for that disparity. This method is supposed to be more robust than
classification .

<figure>
<img src="images/equation1.PNG" id="fig:equation1" />
</figure>

For training purposes intermediate supervision was used with the same
regression predicted disparity, but earlier in the 3D CNN. The training
cost was calculated as a combination of the final cost and the two
intermediate costs.

  [1]: #eq1:regression
 
 
 # Pretrained model

In this section we are trying to reproduce the results in the paper
using the pretrained models downloaded from the [PSMNet Github][]. Due
to resource constraints we could not replicate the situation in which
the author of PSMNet tested its model on Sceneflow or KITTI 2012/2015.
At first we had to switch back to batchsize 1 and 1 worker instead of
the batchsize 12 and 4 workers used for training these models. The End
Point Error (EPE) calculated by us was 4 times higher than in the paper
for the sceneflow test data-set. We found similar errors using the KITTI
2012/2015 pretrained. A hypothesis was drawn that this could be because
of a different batchsize.

In the next experminents we tried to acquire more resources to train
with a higher batchsize. We tried to use Google Cloud, but were not
allowed to allocate multiple GPU’s (The paper used 4 GPU’s; Actually we
did not manage to allocate a single GPU on Google Cloud). We settled
with Google Colab were it was possible possible to train with a
batchsize of 4.

The results of these experiments are plotted below in table [1][]. There
was modest decrease in EPE while training with a higher batchsize. The
EPE is still far away from the 1.19 advertised in the paper and the
batchsize jump from 4 to 12 is not going to bridge that gap.



| Model      |         |      SceneFlow       | Kitti 2015 | Kitti 2012 |
|:-----------|:--------|:--------------------:|:----------:|:----------:|
| Batch size | workers | EPE (End-Point-Error |            |            |
| 1          | 1       |        5.738         |   5.638    |   5.520    |
| 3          | 2       |        5.721         |   5.612    |   5.502    |
| 4          | 2       |        5.701         |     \-     |   5.464    |

Results using pre-trained models



After this disappointment we dove into the code to find any mistakes.
Little did we know there were other reprodrucer that had the same
problem.

## 1.17 factor

We found one of the culprits that had to do with deprecated function
function upsample(). Adding the variable align\_corners=True to those
functions in the original script increased performance, but still it was
far from the paper’s performance.

A second issue that some researchers had trying to use the pre-trained
was an overall decrease in output disparity. As described in this
[Github Issue][] multiplying the output disparity by a factor 1.17
decreased the EPE by a significant margin. In our case this worked as
well and we managed to produce an EPE of 1.52 compared to the 1.19 in
the paper. This 1.17 factor was supposed to only work for the Sceneflow
pre-trained model and we found out that was true in our case as well,
because 1.17 factor did not decrease EPE with the KITTI 2012 pre-trained
model. The cause of this factor is unknown to us, researchers on the
Github page say that by training the model thems

  [PSMNet Github]: https://github.com/JiaRenChang/PSMNet
  [1]: #eq1:regression
  [2]: #tab:results
  [Github Issue]: https://github.com/JiaRenChang/PSMNet/issues/64
  
  
