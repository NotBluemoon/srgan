<a id="readme-top"></a>

# SRGAN – Super Resolution Generative Adversarial Network
Implementation of SRGAN x4 (Super Resolution GAN) faithful to the [2017 SRGAN paper by Ledig et al](https://arxiv.org/abs/1609.04802). 
My goal is to develop a clean PyTorch project that I can use as a reference in the future.

### Built With
* [![PyTorch][pytorch-shield]][pytorch-url]
* [![Python][python-shield]][python-url]
* [![Dataset][div2k-shield]][div2k-url]



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
        <ul>
            <li><a href="#project-structure">Project Structure</a></li>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#dataset">Dataset</a></li>
            <li><a href="#training">Training</a></li>
            <li><a href="#inference">Inference</a></li>
        </ul>
    </li>
    <li>
      <a href="#roadmap">Roadmap</a>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This SRGAN is implemented according to the research paper with minimal changes. Since a lot of work has been done on improving
SRGAN since its introduction, some architecture or parameter choices
may not be the most optimal according to today's standard. The changed choices, along with eplxicit choices mentioned in the paper
are described below.

### Changed 
* ``Train dataset: ImageNet Random 350k subset (BRG) -> DIV2K 800 (RGB)``  
    * Changed to accomodate project requirements, models are trained on RGB instead of BRG
* ``Pretrain steps: 1_000_000 -> 2286``
    * Scaled according to dataset size, not final as the model is not learning enough
* ``Train steps: 200_000 -> 257``
    * Same reasoning as above
* ``Learning rate switch step: 100_000 -> 128``
    * Same reasoning as above, uses floor division of train_steps / 2
* ``Mean-opinion-score (MOS) test metric not used``
    * No human participants involved

### Unchanged
#### Data
* ``Scaling factor = 4``
* ``Crop size: 96 x 96 for HR images, 24 x 24 for LR images``
* ``Range of LR images: [0, 1], HR images: [-1, 1]``

#### Model Architecture
* ``Number of residual blocks B = 16``
* ``Leaky ReLU α = 0.2``

#### Training
* ``Adam β1 = 0.9``
* ``Pretrain learning rate = 0.0001``
* ``Range of LR images: [0, 1], HR images: [-1, 1]``
* ``Content loss: ``
$$
\mathcal{l}_{\text{VGG}/i,j}^{\text{SR}} =
\frac{1}{12.75}
\frac{1}{W_{i,j} H_{i,j}}
\sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}}
\left(
\phi_{i,j}(I^{HR})_{x,y}-
\phi_{i,j}(G_{\theta_G}(I^{LR}))_{x,y}
\right)^2
$$
    * Mean squared loss between the features of reconstructed and reference image, extracted with VGG54 derived from VGG19, then scaled with a factor of 1/12.75 to match pixel loss.
* ``Adversarial loss:``
$$
\mathcal{l}_{\text{adv}} =
\sum_{n=1}^{N}
\left(-\log D_{\theta_D}(G_{\theta_G}(I_n^{LR}))
\right)
$$
    * Achieved using binary cross entropy with target tensor set to 1 and reduction set to sum

#### Test Metrics
All metrics are calculated on the y-channel of center cropped, remove of a 4-pixel wide strip from each border (extracted from SRGAN paper)
* ``Peak Signal-to-Noise Ratio (PSNR)``
* ``Structure Similarity Index (SSIM)``

### Results
To be added as a satisfactory model has not been trained.



<!-- GETTING STARTED -->
## Getting Started
### Project Structure
```
SRGAN/
│
├── data/
│
├── src/
│   └── srgan/
│       ├── data/
│       ├── models/
│   
├── train.py
├── test.py
├── utils.py
├── requirements.txt
└── README.md
```

### Installation
```
git clone https://github.com/NotBluemoon/SRGAN.git
cd SRGAN
pip install -r requirements.txt
```

### Dataset
#### Train Dataset
For this implementation, the DIV2K bicubic subset with a
scale factor `X4` is used to train the network. The SRGAN paper downsamples HR images from the ImageNet database using bicubic kernel with downsampling factor `r = 4`.
```
data/
└── DIV2K/
    ├── DIV2K_train_HR
    ├── DIV2K_train_LR_bicubic

```
#### Test Dataset
Currently the implementation only supports using Set14 with a scale factor of 4 as test dataset, and it should be put in the data folder at project root

### Training
```
python train.py --device auto --batch_size 16 --lr 0.0001 --b1 0.90 --b2 0.999 --checkpoint_interval 5000 --pretrain_steps 100_000 --train_steps 200_000 --num_res_blocks 16
```
Running `python train.py` will use the default parameters which are the ones used in the SRGAN paper.
The only parameter changed from the paper is the pretraining steps and training steps since a different dataset is used.

Note that the `--resume` should only be used if `pretrain_steps` and `train_steps` are unchanged. It should only
be used in case the training was interrupted and no other parameters are changed.

For both `pretrain_steps` and `train_steps`, the value is scaled according 800/350000 to the dataset size used, in which the paper used 350k images
from ImageNet and this implementation uses only 800 images from DIV2K.



### Inference
```
python test.py --image path/to/image.jpg
```
Output will be saved in `results/`.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
- [ ] Train a good SRGAN model
- [ ] Add YAML
- [ ] Merge test.py and infer.py
- [ ] Better logging of metrics during training
- [ ] Logging of test metrics (instead of just printing on terminal)
- [ ] Automatic selection of final model based on best metric during training
- [ ] Generalise functions
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Resources that are used or I have referenced in the development of this project.
* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [PyTorch-GAN GitHub Repo](https://github.com/eriklindernoren/PyTorch-GAN/tree/master)
* [DIV2K Dataset](https://github.com/eriklindernoren/PyTorch-GAN/tree/master)
* [Best-README-Template Github Repo](https://github.com/othneildrew/Best-README-Template/tree/main)
* [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[div2k-shield]: https://img.shields.io/badge/Dataset-DIV2K-blue
[div2k-url]: https://data.vision.ee.ethz.ch/cvl/DIV2K/