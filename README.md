# Schema First! Learn Versatile Knowledge Graph Embeddings by Capturing Semantics with MASCHInE

For the sake of reproducibility, resources for replicating the experiments presented in our paper are provided below.

## Datasets
The ``datasets/`` folder contains the following datasets: ``FB14K``, ``DB77K``, and ``YAGO14K`` [1].

## Building protographs
Two heuristics for building protographs are presented in our paper. In order to build the required protographs for ``FB14K``, ``DB77K``, and ``YAGO14K`` at the same time, please run the following commands:

`python get_prototype.py --dataset FB14K && python get_prototype.py --dataset DB77K && python get_prototype.py --dataset YAGO14K`

Note that you can bring your own datasets (with all the required files) and run the following command:

`python get_prototype.py --dataset mydataset`

## Knowledge Graph Embeddings
Pre-trained embeddings' files are provided in the ``datasets/`` folder. These correspond to the embeddings found at the best epoch on the validation, for each combination of model, setting, and dataset. In particular, for each dataset the ``MASCHInE-P1/`` (resp. ``MASCHInE-P2/``) folder contain embeddings of the best models **after** the fine-tuning step.

We also made our scripts for training and testing available. These will be refactored upon acceptance.
In particular, the ``_vanilla/`` folder contains all the necessary files to train and test knowledge graph embedding models in the vanilla setting. The ``_transfer/`` folder has the same purpose, but for training and testing MASCHInE-P1 and MASCHInE-P2.

## Hyperparameters
Below are reported the best hyperparameters found, which were used for training models:

| YAGO14K  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 100       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 100       | 0.001         | 2048       | L2          | 0.0001             |
| ComplEx  | 100       | 0.01          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 512        | None        | None               |
| TuckER   | 200       | 0.001         | 128        | None        | None               |

| FB14K  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 200       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 200       | 0.001         | 2048       | L2          | 0.01             |
| ComplEx  | 200       | 0.001          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 128        | None        | None               |
| TuckER   | 200       | 0.0005         | 128        | None        | None               |

| DB77K  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 200       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 200       | 0.001         | 2048       | L2          | 0.01             |
| ComplEx  | 200       | 0.001          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 512        | None        | None               |
| TuckER   | 200       | 0.001         | 128        | None        | None               |


## Link Prediction
Link prediction experiments can be replicated using the code provided in the ``_vanilla/`` and ``_transfer/`` folders.

## Entity Clustering
Clustering experiments are performed following the guidelines and code provided in https://github.com/mariaangelapellegrino/Evaluation-Framework.

## Node Classification
Node classification experiments are performed following the guidelines and code provided in https://github.com/janothan/DL-TC-Generator.

## References
[1] Hubert, N., Monnin, P., Brun, A., & Monticolo, D. (2023). [Enhancing Knowledge Graph Embedding Models with Semantic-driven Loss Functions. arXiv preprint arXiv:2303.00286](https://arxiv.org/abs/2303.00286).
