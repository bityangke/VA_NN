## This is the pytorch implement of View Adaptive Neural Networks(VA_NN)

You can also reference to [microsoft official code](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition).

### Prerequisites

* Python 3.6
* PyTorch 1.2
* Opencv 3.4
* Other packages can be found in ```requirements.txt```

### Data Preparation

Firstly, we need to download the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset.

Other Datasets: not supported now

* Extract the dataset to ```./data/NTU-RGB+D/nturgb+d_skeletons/```

* Process the data

```
cd ./data

python ntu_generate_data.py
```

Finally, we get the cross-view and cross-subject subsets for training, containing train, validate and test dataset seperately.

### Train

`python main.py`

### Test

`python main.py --mode test`

### Reference

[paper links](https://arxiv.org/abs/1804.07453)
