# Evaluating the Quality of a Synthesized Motion with the Fréchet Motion Distance
This repository is related to the Fréchet Motion Distance. The FMD measures the distance between the distribution of a ground truth and synthetic motion dataset. It is inspired by the Fréchet Gesture Distance proposed by [1]. The autoencoder model is inspired by [2].

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

## References
[1]
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
[2]
```
https://alanbertl.com/autoencoder-with-fast-ai/
```
