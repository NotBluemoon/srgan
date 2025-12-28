<a id="readme-top"></a>

# SRGAN – Super Resolution Generative Adversarial Network
Implementation of SRGAN (Super Resolution GAN) faithful to the [2017 SRGAN paper by Ledig et al](https://arxiv.org/abs/1609.04802). 
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



<!-- GETTING STARTED -->
## Getting Started
### Project Structure
```
SRGAN/
│
├── data/
│   ├── HR/
│   └── LR/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── vgg.py
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
For this implementation, the DIV2K bicubic subset with a
scale factor `X4` is used to train the network and can be obtained here. The SRGAN paper downsamples HR images from the ImageNet database using bicubic kernel with downsampling factor `r = 4`.
```
data/
└── DIV2K/
    ├── DIV2K_train_HR
    ├── DIV2K_train_LR_bicubic
    ├── DIV2K_valid_HR
    └── DIV2K_valid_LR_bicubic
```

### Training
```
python train.py --device auto --batch_size 16 --lr 0.0001 --b1 0.90 --b2 0.999 --checkpoint_interval 5000 --pretrain_steps 100_000 --train_steps 200_000 --num_res_blocks 16
```
Running `python train.py` will use the default parameters which are the ones used in the SRGAN paper.
Feel free to change the training parameters according to your current needs.

### Inference
```
python test.py --image path/to/image.jpg
```
Output will be saved in `results/`.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
- [ ] Add YAML
- [ ] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish
  
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