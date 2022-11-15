# rtb-mtae
This is a repository of the experimental code supporting the paper Multi-task Learning for CTR Prediction and Market Price Modeling in Real-time Bidding Advertising. The source code and datasets will be released when the paper is published.

This repository is the implementation supporting our CIKM 2021 paper entitled "Multi-task Learning for CTR Prediction and Market Price Modeling in Real-time Bidding Advertising". The final version of this paper has been published on CIKM 2021: https://dl.acm.org/doi/abs/10.1145/3459637.3482373. For any problem, please feel free to concact the author Haizhi Yang (sehzyang@mail.scut.edu.cn).

## Abstract
The rapid rise of real-time bidding-based online advertising has brought significant economic benefits and attracted extensive research attention. From the perspective of an advertiser, it is crucial to perform accurate utility estimation and cost estimation for each individual auction in order to achieve cost-effective advertising. These problems are known as the click through rate (CTR) prediction task and the market price modeling task, respectively. However, existing approaches treat CTR prediction and market price modeling as two independent tasks to be optimized without regard to each other, thus resulting in suboptimal performance. Moreover, they do not make full use of unlabeled data from the losing bids during estimations, which makes them suffer from the sample selection bias issue. To address these limitations, we propose Multi-task Advertising Estimator (MTAE), an end-to-end joint optimization framework which performs both CTR prediction and market price modeling simultaneously. Through multi-task learning, both estimation tasks can take advantage of knowledge transfer to achieve improved feature representation and generalization abilities. In addition, we leverage the abundant bid price signals in the full-volume bid request data and introduce an auxiliary task of predicting the winning probability into the framework for unbiased learning. Through extensive experiments on two large-scale real-world public datasets, we demonstrate that our proposed approach has achieved significant improvements over the state-of-the-art models under various performance metrics.

## Experiment Setting
### Python Setting
Recommended version:  
* python (>=3.7) 
* tensorflow (>=2.3)
* lightgbm (=2.3, for truthful bidder)

### Sample Data
Here, we upload the data of campaign 1458 in iPinYou Dataset as an example. You can download the data from the following link and put the files in /data/ipinyou/1458/.

Download link: https://pan.baidu.com/s/1ZGs4Ord4JLCjcB_aUgeWTg

Extraction code: w2cw

### Data Preparation
First step: Update the root path in config/base_config.py according to your own absolute path.
```
root_path = '/root/rtb-mtae/'
```

Second step: Preprocess the data and encode the features.
```
python src/util/processor.py
```

Third step: Train a CTR baseline model (lightgbm) for the truthful bidder. The truthful bidder is used to simulate bidding and split the original training logs into winning dataset and losing dataset.
```
python src/util/truthful_bidder.py
```

## Run MTAE
Please first preprocess the data and run the following code to train MTAE.
```
python src/mt_model/mtae_train.py
```

## Citation
```
@inproceedings{yang2021multi,
  title={Multi-task Learning for Bias-Free Joint CTR Prediction and Market Price Modeling in Online Advertising},
  author={Yang, Haizhi and Wang, Tengyun and Tang, Xiaoli and Li, Qianyu and Shi, Yueyue and Jiang, Siyu and Yu, Han and Song, Hengjie},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2291--2300},
  year={2021}
}
```