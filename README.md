## <p align="center">Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates<br><br>ICML 2023<br><br></p>

<div align="center">
  <a href="https://louissharrock.github.io/" target="_blank">Louis&nbsp;Sharrock</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://chris-nemeth.github.io/" target="_blank">Christopher&nbsp;Nemeth</a> &emsp; </b>
</div>

## Description

This repository contains code to implement the coin sampling algorithms 
described in [Sharrock et al. (2023)](https://arxiv.org/abs/2301.11294). The 
basic implementation of the algorithms - Coin SVGD, Coin LAWGD, and Coin KSDD - 
can be found in ``main.py``. 

## Experiments

The results in the paper can be reproduced using the various python notebooks.

|        File        | Example                                              |
|:------------------:|:-----------------------------------------------------|
| ``toy_svgd.ipynb`` | Toy examples                                         |
|    Coming soon.    | Bayesian independent component analysis              |
|    Coming soon.    | Bayesian logistic regression                         |
|    Coming soon.    | Bayesian neural network                              |
|    Coming soon.    | Bayesian probabilistic matrix factorisaton           |

## Citation

If you find the code in this repository useful for your own research, 
please consider citing our paper:

```bib
@InProceedings{Sharrock2023,
  title = 	 {Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates},
  author =       {Sharrock, Louis and Nemeth, Christopher},
  booktitle = 	 {Proceedings of The 40th International Conference on Machine Learning},
  year =         {2023},
  city =         {Honolulu, Hawaii},
}
```

## Acknowledgements

Our implementations of Coin SVGD, Coin LAWGD, and Coin KSDD, are based on existing 
implementations of SVGD, LAWGD, and KSDD. We gratefully acknowledge the authors
of the following papers for their open source code:
* Q. Liu and D. Wang. Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm. NeurIPS, 2016. [[Paper](https://arxiv.org/abs/1608.04471)] | [[Code](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent)].
* S. Chewi, T. Le Gouic, C. Lu, T. Maunu, P. Rigollet. SVGD as a kernelized Wasserstein gradient flow of the chi-squared divergence. NeurIPS, 2020. [[Paper](https://arxiv.org/abs/2006.02509)] | [[Code](https://github.com/twmaunu/LAWGD)].
* A. Korba, P.-C. Aubin-Frankowski, S. Majewski, P. Ablin. ICML 2021. [[Paper](https://arxiv.org/abs/2105.09994)] | [[Code](https://github.com/pierreablin/ksddescent)].

