# Neural-Style-Transfer-Gatys
This is a Tensorflow implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. 

### Some details

* The pretrained VGG19 model comes from this [repo](https://github.com/machrisaa/tensorflow-vgg).  
* Some implementations are inspired by [Neural Transfer with PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)  

### Quick Start
```
python3 transfer_gatys_tf.py --content_img content_img.jpg --style_img style_img.jpg --lr_rate 1 --epoch 3000
```
View the code for more details about arguments.

### Results
<div>
<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/content_img.jpg" width="284.8" height="189.6">
<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/out_3000.jpg" width="284.8" height="189.6">
<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/style_img.jpg" width="284.8" height="189.6">
    

<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/img/zju.jpg" width="49%" height="49%">
<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/img/starry.jpg" width="49%" height="49%">

<img src="https://github.com/VainF/Neural-Style-Transfer-Gatys/blob/master/img/zju_5000.png" width="98%" height="98%">
</div>
