# Schema First! Learn Versatile Knowledge Graph Embeddings by Capturing Semantics with MASCHInE

For the sake of reproducibility, resources for replicating the experiments presented in our paper are provided below.

## Datasets
The ``datasets/`` folder contains the following datasets: ``YAGO14K``, ``FB15k187``, and ``DBpedia77k`` [1].


| Dataset    |       | $|\mathcal{E}|$ | $|\mathcal{R}|$ | $|\mathcal{T}|$ |
|------------|-------|-----------------|-----------------|-----------------|
| YAGO14k    | KG    | 14,178          | 37              | 19,183          |
|            | P1    | 22              | 37              | 37              |
|            | P2    | 590             | 37              | 4,959           |
| FB15k187   | KG    | 14,305          | 187             | 278,436         |
|            | P1    | 138             | 187             | 187             |
|            | P2    | 138             | 187             | 187             |
| DBpedia77k | KG    | 76,651          | 150             | 190,028         |
|            | P1    | 55              | 150             | 150             |
|            | P2    | 186             | 150             | 3,210           |



## Building Protographs
Two heuristics for building protographs are presented in our paper. In order to build the required protographs for ``YAGO14K``, ``FB15k187`` (renamed as ``FB14K`` for short), and ``DBpedia77k`` (``DB77K`` for short) at the same time, please run the following commands:

`python get_prototype.py --dataset YAGO14K && python get_prototype.py --dataset FB14K && python get_prototype.py --dataset DB77K`

Note that you can bring your own datasets (with all the required files) and run the following command:

`python get_prototype.py --dataset mydataset`

## Knowledge Graph Embeddings
Pre-trained embeddings' files are provided in the ``datasets/`` folder. These correspond to the embeddings found at the best epoch on the validation, for each combination of model, setting, and dataset. In particular, for each dataset the ``MASCHInE-P1/`` (resp. ``MASCHInE-P2/``) folder contain embeddings of the best models **after** the fine-tuning step.

We also made our scripts for training and testing available. These will be refactored upon acceptance.
In particular, the ``_vanilla/`` folder contains all the necessary files to train and test knowledge graph embedding models in the vanilla setting. The ``_transfer/`` folder has the same purpose, but for training and testing MASCHInE-P1 and MASCHInE-P2. Before using these scripts, you should first place them at the root of this repo (i.e. in their parent folder).

## Hyperparameters
Below are reported the best hyperparameters found, which were used for training models:

| YAGO14K  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 100       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 100       | 0.001         | 2048       | L2          | 0.0001             |
| ComplEx  | 100       | 0.01          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 512        | None        | None               |
| TuckER   | 200       | 0.001         | 128        | None        | None               |

| FB15k187  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 200       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 200       | 0.001         | 2048       | L2          | 0.01             |
| ComplEx  | 200       | 0.001          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 128        | None        | None               |
| TuckER   | 200       | 0.0005         | 128        | None        | None               |

| DBpedia77K  | dimension | learning rate | batch size | regularizer | regularizer weight |
|----------|-----------|---------------|------------|-------------|--------------------|
| TransE   | 200       | 0.001         | 2048       | L2          | 0.001              |
| DistMult | 200       | 0.001         | 2048       | L2          | 0.01             |
| ComplEx  | 200       | 0.001          | 2048       | L2          | 0.1                |
| ConvE    | 200       | 0.001         | 512        | None        | None               |
| TuckER   | 200       | 0.001         | 128        | None        | None               |


## Link Prediction
Link prediction experiments can be replicated using the code provided in the ``_vanilla/`` and ``_transfer/`` folders.

## Entity Clustering
Clustering experiments are performed following the guidelines and code provided in https://github.com/mariaangelapellegrino/Evaluation-Framework [2].

## Node Classification
Node classification experiments are performed following the guidelines and code provided in https://github.com/janothan/DL-TC-Generator [3].

## References
[1] Hubert, N., Monnin, P., Brun, A., & Monticolo, D. (2023). [Treat Different Negatives Differently: Enriching Loss Functions with Domain and Range Constraints for Link Prediction](https://arxiv.org/abs/2303.00286).

[2] Pellegrino, M. A., Cochez, M., Garofalo, M., & Ristoski, P. (2019). [A configurable evaluation framework for node embedding techniques.](https://link.springer.com/chapter/10.1007/978-3-030-32327-1_31) In The Semantic Web: ESWC 2019 Satellite Events: ESWC 2019 Satellite Events, Portorož, Slovenia, June 2–6, 2019, Revised Selected Papers 16 (pp. 156-160). Springer International Publishing.

[3] Portisch, J., & Paulheim, H. (2022, October). [The DLCC node classification benchmark for analyzing knowledge graph embeddings.](https://arxiv.org/abs/2207.06014) In The Semantic Web–ISWC 2022: 21st International Semantic Web Conference, Virtual Event, October 23–27, 2022, Proceedings (pp. 592-609). Cham: Springer International Publishing.
