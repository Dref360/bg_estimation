# Background estimation using deep learning (Not maintained)

Input            |  Output
:-------------------------:|:-------------------------:
![](/imgs/gt_output.gif)  |  ![](/imgs/unetoutput.gif)

This repo has 4 models to estimate the background of a video.

VGG3D, UNET, VAE (not working), SingleFrame (not using video)

The database used is from : http://scenebackgroundmodeling.net/

I recommend being on Keras master branch since there is a lot of fixes.

This repo is using Tensorflow, it would not works using Theano since the SSIM is not implemented using the Keras Backend.


## Goal

The goal is that from a video, we extract the background

![](/imgs/goal.png)

## VGG+

I tested many models, but the best seems to be VGG+, a VGG3D implementation with a skip-connection.

![](/imgs/vggplus.png)

For the loss, I used SSIM with an MSE.


![](/imgs/loss.png)

Here's a comparison between VGG3D and VGG+

<table>
  <tr>
    <td>VGG3D</td>
    <td>VGG+</td>
  </tr>
  <tr>
    <td colspan="2"><img src='/imgs/diff.png'/></td>
  </tr>
</table>


# Examples
![](/imgs/ex.png)


## Install Tensorflow

Refer to this page to be kept up to date!
https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#virtualenv-installation
This repo is using tf 0.11.0rc

### Install cv3 on Ubuntu

http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
