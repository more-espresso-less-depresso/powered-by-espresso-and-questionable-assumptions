# Repository Structure
The repository consists of a set of scripts for running experiments, mainly the files `all_samples.py` and `grouped_samples.py`. The raw datasets are available in the `datasets`folder, and the previous results are stored in the `results` folder, with separate folder per model and dataset used for testing. There exists three notebooks, which are used to generate visualisations, response statistics and dataset statistics.

# Running the project
You run the project for the different models given by functions in the `models.py` file by either selecting to run the `all_samples.py` or `grouped_samples.py` file. The `all_samples.py` file will generate activations and responses to benign and harmful prompts separately, whereas the `grouped_samples.py` generates activations, responses and response classifications for the combined set of benign and harmful prompts. For each of the files, the fields:
```
MODEL_PATH = "<model-identifier>"
DATASET_PATH = "<dataset-identifier>"
DATASET = <dataset-loader>
MODEL = <model-loader>
```
Determine the dataset and model to use for the experiments, and these should be set accordingly. The `<dataset-loader>` can be any of the functions present in the `dataset_loaders.py`, and the `<model-loader>` can be any of the functions present in the `models.py`.

After your experiments have finished, you can generate stats for the response classifications, and visualise the activations in the notebooks. For this you will need to change these fields to target a specific model and dataset:
```
model_path = "<model-identifier>"
dataset_path = "<dataset-identifier>"
```