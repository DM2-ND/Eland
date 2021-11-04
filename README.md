Action Sequence Augmentation for Early Graph-based Anomaly Detection
====
This repository contains the source code for the CIKM'2021 paper:

[Action Sequence Augmentation for Early Graph-based Anomaly Detection](https://arxiv.org/pdf/2010.10016.pdf)

by [Tong Zhao*](https://tzhao.io/) (tzhao2@nd.edu), [Bo Ni*](https://arstanley.github.io/) (bni@nd.edu),  [Wenhao Yu](https://wyu97.github.io/), [Zhichun Guo](), [Neil Shah](http://nshah.net/), and [Meng Jiang](http://www.meng-jiang.com/).

## Requirements

This code package was developed and tested with Python 3.8.5. Make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Example Usage

```
python -u main.py --dataset reddit --rnn gru --method gcn --graph_num 3
```

## Cite
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{zhao2021action,
  title={Action Sequence Augmentation for Early Graph-based Anomaly Detection},
  author={Zhao, Tong and Ni, Bo and Yu, Wenhao and Guo, Zhichun and Shah, Neil and Jiang, Meng},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={},
  year={2021}
}
```

