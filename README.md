## :sparkles: Oriented Object Detection (OBB) Competition Solution.
Finalist's solution in the track of Oriented Object Detection in Remote Sensing Images, 2022 Guangdong-Hong Kong-Macao Greater Bay Area International Algorithm Competition.  
## :hammer: Installation
This project is based on [Jitto](https://github.com/Jittor/jittor) framework. Please follow the official installation [documentation](https://github.com/HsLOL/JDET/blob/master/JDET_README.md) for installation.
## :busts\_in\_silhouette: Team Members (Random Ranking)
[Jianhong Han](https://github.com/HsLOL), [Zhonghao Fang](https://github.com/HsLOL), [Zhaoyi Luo](https://github.com/HsLOL)  
## :bulb: Features
- Backbone  
- [x] Support Swin-Transformer Tiny/Small/Base/Large Backbone Network.
- Neck  
- [x] Support PAFPN network.
- Optimizer
- [x] Support AdamW Optimizer.
- Some Useful Tools  
- [x] Support Model Ensemble.
- [x] Support Soft-NMS, Class-Agnostic NMS.
- [x] Support HSV Data Augmentation.
## :pushpin: Solutions  
- Training Data Augmentation  
We use random combination of hsv, horizontal/vertical flip, rotation for data augmentation.  
- Multi-scale training and testing
The training images are scaled to 0.5,1,1.5 times and cropped to 1024x1024 for training and testing.  
- Swin Transformer Backbone  
We use Swin-Transformer as backbone in Oriented R-CNN, S^2ANet and ROI Transformer for better performance.  
- Model Ensemble  
We merge the detection results from Oriented R-CNN, S^2ANet and ROI Transformer for better performance.
- Test Time Augmentation
We use extra random horizontal/vertical flip, random rotation for inference phrase.  
- Soft NMS and Class-Agnostic NMS
We use Class-Agnostic NMS for post-processtion. Soft-NMS used but not work.  
## :tada: Visualization
