# Parallel XAI GNN Neighbors

This is the official implementation of [ParaParallelizing Node-Level Explainability in Graph Neural Networks](https://arxiv.org/abs/2403.16108).

In this README you will find as much information as possible so you be able to replicate this research. If you have doubts or queries about it, please email us, and we will try to help as much as possible. 

## Code structure

In this section we will explain what you can fnd in the different files that we have in this repository.

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

- main.py: This will be the file executed to compute the different experiments. Here, you may choose between three types of executions:
    1. examples: This mode will execute the `main` function of the `experiments/example.py` file. This will generate a couple of visualizations of explainability technique for node classification with a GNN.
    2. full_tables: This mode will execute the `main` function of the `experiments/full_tables.py` file. This will generate results, as tables and visualizations, for the full reconstruction method.
    3. drop_tables: This mode will execute the `main` function of the `experiments/drop_tables.py` file. This will generate results, as tables and visualizations, for the dropout reconstruction method.

- experiments folder:
    - 

## Dependencies

They are listed in the pyproject.toml file. For installing them, you will need uv. Just by running the following command, you can recreate our environment:

```
uv sync
```

## Cite

Please cite our [paper](https://arxiv.org/abs/2403.16108) if you find it useful:


```
@misc{gonzalez2024transformer,
      title={Parallelizing Node-Level Explainability in Graph Neural Networks}, 
      author={Oscar Llorente Gonzalez},
      year={2024},
      eprint={2403.16108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




