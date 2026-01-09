# Parallel XAI GNN Neighbors

This is the official implementation of [ParaParallelizing Node-Level Explainability in Graph Neural Networks](https://arxiv.org/abs/2601.04807).

In this README you will find as much information as possible so you be able to replicate this research. If you have doubts or queries about it, please email us, and we will try to help as much as possible. 

## Code structure

In this section we will explain what you can fnd in the different files that we have in this repository. It is important to clarify that all teh paths that will be used for storing results or data will be created if thet don't exist.

### src module

All our code is saved under the src folder. Here we can find two folders:

- train: Here we save all the code that is meant for training our models.
- explain: All the code related to the explainability of the models.

Besides these two folders, we also have the utils.py file. There we save auxiliary functions used in both modules, train and explain.

### Train module

Inside the `train` folder, we have all the code for training the models. In this case, we use a two layer GNN and a final linear layer. The models are using GCN or GAT. We can find different files in the folder:

- models.py: Here one can find the definition of the models.

- train.py: This file is the one executed to train the models. When being executed, it will train all the models and save them under the `models` folder (it will be created if it doesn't exist). The file must be executed with the following command from the root directory:

      python -m src.train.train

### Explain module

The `explain` folder is the one that contains most of the code for the present research. Here you can find several files:

- main.py: This will be the file executed to compute the different experiments. It will be executed with the following command from the root directory.

      python -m src.explain.main

   You may choose between three types of executions by changing the `experiment_name` in the main function:

    1. examples: This mode will execute the `main` function of the `experiments/example.py` file. This will generate a couple of visualizations of explainability techniques for node classification with a GNN.
    2. full_tables: This mode will execute the `main` function of the `experiments/full_tables.py` file. This will generate results, as tables and visualizations, for the full reconstruction method.
    3. drop_tables: This mode will execute the `main` function of the `experiments/drop_tables.py` file. This will generate results, as tables and visualizations, for the dropout reconstruction method.

- experiments folder:
    - examples.py: This file generates two example visualizations of explainability for a GNN in a node classification problem. They will be stored inside the `results/images` folder. Visualizations will be for the `Saliency Map` technique and `GNNExplainer`. They will be stored in pdf files.
    - full_tables.py: This file computes the results for the full reconstruction method. Results will be saved inside the `results/full` folder. There you will find two folders, `tables` and `charts`. `tables` folder stores all the dataframes with the results in csv and latex format. `charts` contains charts comparing the evolution of the graph size and the execution time.
    - drop_tables.py: This file computes the results for the dropout reconstruction method. Results will be saved inside the `results/drop` folder. Here we have only created the dataframes, which will also be in csv and latex formats. 

- executions.py: This file contains the code for the different executions to compute the explainability. Here we can find the following functions:
    1. compute_xai: This is a generic function to compute the explainability with the proper execution and store the time and the results. Specifically, this will execute the `original_xai` function if the number of clusters is 1, and `parallel_xai` if it exceeds 1. It will execute each technique 3 times and then take the average of these three executions. Since several computations are repeated throughout the different experiments, this function will create a `checkpoints` folder and store the sparse matrices with the time of execution. This folder will weigh around 1.3 GB when all the results are computed. 
    2. original_xai: In this function, we compute the original explainability, sequentially node by node. 
    3. parallel_xai: This function implements the parallel computation we have developed in the paper.

- cluster.py: In this file, the creation of `data_extended`, the dataset with the reconstructed edges.


- methods.py: This file has the code to implement the different explainability techniques used in the paper.

- sparse.py: In this file is stored the code to implement a sparse matrix to save the explainability for all the nodes. This matrix has dimensions [number of nodes, number of nodes]. We have used scipy to implement it. It is based on lil matrices.

- utils.py: This file contains the code related to auxiliary functionality for the explain module.

*Note: due to the size of the graph once reconstructed, in the case of the `PubMed` dataset, it will only be executed until 32 clusters for GCN and 16 for GAT.*

## Use this research

In this section we will try to explain which is the easier way to use the research that we have published here in your experiments or executions. The important function that contains all the logic is `parallel_xai` inside `src/explain/executions.py`. You can copy that function and all the others it calls to run if you only want to use the parallelization technique we developed here.

## Dependencies

They are listed in the pyproject.toml file. For installing them, you will need uv. Just by running the following command, you can recreate our environment:

```
uv sync
```

## Makefile

We have included this file just to run some checks offline to improve the quality of our code, but you don not have to worry about this file to reproduce the code. The checks are the following ones:

- Black

- Mypy

- Flake8

- Pylint

Specific rules for the checks are included in our pyproject.toml file.

## Cite

Please cite our [paper](https://arxiv.org/abs/2601.04807) if you find it useful:


```
@misc{llorenteParallel,
      title={Parallelizing Node-Level Explainability in Graph Neural Networks}, 
      author={Llorente, Oscar and Boal, Jaime and  S{\'a}nchez-{\'U}beda, Eugenio F and Diaz-Cano Antonio and Familiar-Cabero},
      year={2026},
      eprint={2601.04807},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




