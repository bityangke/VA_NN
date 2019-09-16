**This is the pytorch implement of View Adaptive Neural Networks(VA_NN)**

You can also reference to [microsoft official code](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition).

#### Prerequisites

* Python3.6
* PyTorch1.2
* Opencv3.4

#### Data Preparation

Firstly, we need to download the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset, and extract the dataset to ./data/NTU-RGB+D/nturgb+d_skeletons/

Then process the data

'''
cd ./data
python ntu_generate_data.py
'''

Finally, we get the cross-view and cross-subject subsets.

#### Train

'python main.py'

#### Test

'python main.py --mode test'

#### Reference

[paper links](https://arxiv.org/abs/1804.07453)
