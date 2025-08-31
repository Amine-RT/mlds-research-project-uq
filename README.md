# Uncertainty Quantification (UQ) with Auxiliary Networks

This repository contains the code and Collab notebooks for the MSc research project "Uncertainty Quantification with Auxiliary Networks." The project explores the use of auxiliary networks to quantify and manage uncertainty in machine learning models, introducing a new loss function defined as
$$L_{new} = \frac{\sum_{i=1}^{N} \ell(f(x_i), y_i)}{\sum_{j=1}^{N} w_j(x_j)}$$

The repository is structured into three main phases, each corresponding to a distinct stage of the project's development.

### Repository structure

* `Notebook_01/`: Contains the notebook and associated scripts for **Phase 1: Initial Setup and Baseline Modeling**.
* `Notebook_02/`: Contains the notebook for **Phase 2: Classification (proposed approach with $L_{new}$ and ConfidNet**.
* `Notebook_03/`: Contains the notebook for **Phase 2: Selective Classification with SelectiveNet**.

### How to use the Notebooks in Google Colab

To use these notebooks, you can open them directly in Google Colab. This allows you to run the code in a cloud environment without needing to install anything locally.

1.  **Open the notebook in Colab**: Click on the Colab badge for the notebook you want to use.

    * **Phase 1: Initial Setup and Baseline Modeling**
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amine-rt/mlds-research-project-uq/blob/main/Notebook_01/MLDS_research_project_Notebook_01_Phase_1.ipynb)
    * **Phase 2: Classification**
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amine-rt/mlds-research-project-uq/blob/main/Notebook_02/MLDS_research_project_Notebook_02_Phase_2_(Classification).ipynb)
    * **Phase 3: Selective Classification with SelectiveNet**
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amine-rt/mlds-research-project-uq/blob/main/Notebook_03/MLDS_research_project_Notebook_03_Phase_2_(Classification+SelectiveNet).ipynb)

2.  **Mount your Google Drive (Optional)**: Copy the pre-trained model and use the provided code in Notebook 2 or 3 to mount and copy the models into the virtual environement

3.  **Run the Notebook**: Execute the cells in the notebook sequentially. The notebooks automatically access the necessary helper scripts (`data_gen.py`, `f_eval.py`, `f_models.py`, `f_plots.py`, `f_train.py`) from the repository, as the entire repository is cloned into the Colab environment.


### Notebook Descriptions

* **`MLDS_research_project_Notebook_01_Phase_1.ipynb`**: This notebook covers the initial phase of the project, including data generation, model setup, and the training of baseline models to establish a performance benchmark.

* **`MLDS_research_project_Notebook_02_Phase_2_(Classification).ipynb`**: This notebook is the main part of the code that implement the proposed approach of the new loss and compare it with **ConfidNet** implementation

* **`MLDS_research_project_Notebook_03_Phase_2_(Classification+SelectiveNet).ipynb`**: This final notebook is not finalised and implements the **SelectiveNet** approach for selective classification.

### Citation

The notebook implementation was based on the original papers that introduced the core concepts used in this project:

* **SelectiveNet**: A deep neural network with an integrated reject option.
    ```
    @inproceedings{
    geifman2019selectivenet,
    title={SelectiveNet: A Deep Neural Network with an Integrated Reject Option},
    author={Yonatan Geifman and Ran El-Yaniv},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={2151--2159},
    year={2019}
    }
    ```

* **ConfidNet**: Addressing Failure Prediction by Learning Model Confidence.
    ```
    @article{
    corbiere2019addressing,
    title={Addressing Failure Prediction by Learning Model Confidence},
    author={Corbiere, Charles and Thome, Nicolas and Bar-Hen, Avner and Cord, Matthieu and Perez, Patrick},
    journal={arXiv preprint arXiv:1910.04851},
    year={2019}
    }
    ```
  
