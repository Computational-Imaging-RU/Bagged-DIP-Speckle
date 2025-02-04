# Bagged Deep Image Prior for Recovering Images in the Presence of Speckle Noise

## Preview
#### There are 4 python files in this repo. The test data is under the /data/ folder.

- train_BaggedDIP_speckle.py: training Bagged-DIP based PGD algorithm.

- function_grad.py: i) explicit gradient function of the MLE loss function, and ii) implementation of Newton Schulz algorithm for efficently approximating matrix inverse.

- decoder.py: basic network structures of the Deep Image Prior/Deep Decoder we use in Bagged-DIP.

- utils.py: all the other helper functions.

## Running the code

#### Run the Bagged-DIP PGD algorithm for recovering images:

```
python train_BaggedDIP_speckle.py
```

#### Specify the hyperparameters and experiment setting:

#### E.g., recover images from L=100, m/n=0.5 down-sampled complex-valued measurements:

```
python train_BaggedDIP_speckle.py --compression_rate 0.5 --patch_size 128 --lamb 1.0 --crop True --lr_NN 1e-3 --lr_GD 0.01 --outer_ite 100 --num_look 100 --weight_decay 0.0 --test_name 'Set11' --DIP_avg True --use_complex True
```

## Relevant works on speckle noise

[1] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Novel approach to coherent imaging in the presence of speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2024. Vol. 13149. SPIE, 2024. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13149/1314908/Novel-approach-to-coherent-imaging-in-the-presence-of-speckle/10.1117/12.3027824.full)

[2] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Bagged Deep Image Prior for Recovering Images in the Presence of Speckle Noise." Forty-first International Conference on Machine Learning (ICML 2024). [paper](https://openreview.net/pdf?id=IoUOhnCmlX)

[3] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook compressive sensing in the presence of speckle noise." In NeurIPS 2023 Workshop on Deep Learning and Inverse Problems. 2023. [paper](https://openreview.net/forum?id=G8wMnihF6E)

[4] Zhou, Wenda, Shirin Jalali, and Arian Maleki. "Compressed sensing in the presence of speckle noise." IEEE Transactions on Information Theory 68.10 (2022): 6964-6980. [paper](https://ieeexplore.ieee.org/abstract/document/9783054)
