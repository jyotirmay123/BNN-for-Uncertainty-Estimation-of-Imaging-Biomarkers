

# Bayesian Neural Networks for Uncertainty Estimation of Imaging Biomarkers

**Accepted in MICCAI-MLMI-2020**

More details here: [MICCAI-MLMI-2020](https://arxiv.org/pdf/2008.12680.pdf)

Folder Structures:
- *dataset_groups:* It holds various datasets with their respective processing code in it.
- *projects:* It holds various bayesian architectures i.e. 4 that we used for our experiments. Fully Bayesian, Quicknat with dropout, Probabilistic U-Net, Hierarchical U-Net.
- *interface:* It has all the base class for solver, data processing pipeline, evaluator and run setup. Provides a consistent platform for all projects to train and evaluate.
- *utils:* To have utility functions like logger & notifier to mention a few.
- *stat_analysis:* This folder contains post segmentation data analysis with diesease classification and group analysis stuff using python and R toolkit.

if you like the paper, and willing to extend the work, please cite:

```
@inproceedings{senapati2020bayesian,
  title={Bayesian Neural Networks for Uncertainty Estimation of Imaging Biomarkers},
  author={Senapati, Jyotirmay and Roy, Abhijit Guha and P{\"o}lsterl, Sebastian and Gutmann, Daniel and Gatidis, Sergios and Schlett, Christopher and Peters, Anette and Bamberg, Fabian and Wachinger, Christian},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={270--280},
  year={2020},
  organization={Springer}
}
```

## Code Authors

* **Jyotirmay Senapati**  - [jyotirmay-senapati](https://www.linkedin.com/in/jyotirmay-senapati/)