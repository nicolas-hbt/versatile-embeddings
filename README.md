# Schema First! Learn Versatile Knowledge Graph Embeddings by Capturing Semantics with MASCHInE

For the sake of reproducibility, below are provided the necessary resources for replicating the experiments presented in our paper.

## Datasets
The ``datasets/`` folder contains the following datasets: ``FB14K``, ``DB77K``, and ``YAGO14K``. These are the filtered versions of  ``FB15K-237``, ``DB93K``, and ``YAGO4-19K``, respectively [1].

## Building protographs
Two heuristics for building protographs are presented in our paper. In order to build the required protographs for ``FB14K``, ``DB77K``, and ``YAGO14K`` at the same time, please run the following commands:

`python get_prototype.py --dataset FB14K && python get_prototype.py --dataset DB77K && python get_prototype.py --dataset YAGO14K`

Note that you can bring your own datasets (with all the required files) and run the following command:

`python get_prototype.py --dataset mydataset`

## Embeddings
Pre-trained embeddings' files are provided in the ``datasets/`` folder. These correspond to the embeddings found at the best epoch on the validation, for each combination of model, setting, and dataset. In particular, for each dataset the ``MASCHInE-P1/`` (resp. ``MASCHInE-P2/``) folder contain embeddings of the best models **after** the fine-tuning step.

## Entity Clustering
Clustering experiments are performed following the guidelines and code provided in https://github.com/mariaangelapellegrino/Evaluation-Framework.

## Node Classification
Node classification experiments are performed following the guidelines and code provided in https://github.com/janothan/DL-TC-Generator.

## References
[1] Hubert, N., Monnin, P., Brun, A., & Monticolo, D. (2023). [Sem@K: Is my knowledge graph embedding model semantic-aware?] (https://arxiv.org/abs/2301.05601)
