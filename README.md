# Evaluating the Quality of a Synthesized Motion with the Fréchet Motion Distance
This repository is related to the Fréchet Motion Distance and is related to [1]. This work has been presented at the poster session at SIGRAPPH 2022. The FMD measures the distance between the distribution of a ground truth and synthetic motion dataset. It is inspired by the Fréchet Gesture Distance proposed by [2]. The autoencoder model is inspired form [here](https://alanbertl.com/autoencoder-with-fast-ai/).

## Requirements
This implementation has been tested with 
- python 3.7
- fastai 2.5.3
- pytorch 1.10.1
## Usage
To train the network with 34-frames motion using pretrained Enocder with Imagenet
```
python Conv2d.py --norm_image --training --n_poses 34 --all_joints
```
To evaluate, just remove the training flag
```
python Conv2d.py --norm_image --n_poses 34 --all_joints
```
You need to navigate into Inception folder to evaluate your motion dataset with InceptionV3 architecture. Run the following command to measure FMD on the given dataset polluted by the defined noise (--method and --std)
```
python inception_fmd.py --batch_size 512 --dataset h36m --method gaussian_noise --std 0.003
```
## References
[1]
```
@misc{https://doi.org/10.48550/arxiv.2204.12318,
  doi = {10.48550/ARXIV.2204.12318},
  url = {https://arxiv.org/abs/2204.12318},
  author = {Maiorca, Antoine and Yoon, Youngwoo and Dutoit, Thierry},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Evaluating the Quality of a Synthesized Motion with the Fréchet Motion Distance},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
[2]
```
@article{Yoon2020Speech,
  title={Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity},
  author={Youngwoo Yoon and Bok Cha and Joo-Haeng Lee and Minsu Jang and Jaeyeon Lee and Jaehong Kim and Geehyuk Lee},
  journal={ACM Transactions on Graphics},
  year={2020},
  volume={39},
  number={6},
}
``` 
