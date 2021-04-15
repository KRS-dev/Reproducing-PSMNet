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

$$\\hat{d} = \\sum\_0^{D\_{max}} d\*softmax(-c\_d)
    \\label{eq1:regression}$$

For training purposes intermediate supervision was used with the same
regression predicted disparity, but earlier in the 3D CNN. The training
cost was calculated as a combination of the final cost and the two
intermediate costs.

  [1]: #eq1:regression
  
  
  
