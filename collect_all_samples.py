import torch
import random
import logging

import numpy as np

from datasets import load_dataset

from models import *
from activations import *
from visualisations import *
from utils import *
from dataset_loaders import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

MODEL_PATH = "small_mistral"
DATASET_PATH = "wildjailbreak"
DATASET = load_wildjailbreak_dataset
MODEL = small_mistral
RESULT_FOLDER = f"results/{MODEL_PATH}/{DATASET_PATH}/all"
TEMP_PATH = f"temp_activations_{MODEL_PATH}_{DATASET_PATH}_all"

B_ALL_KEY = "b_all"
B_CAT_KEY = "b_cat"
H_ALL_KEY = "h_all"
H_CAT_KEY = "h_cat"

def b_sampling(b_samples: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int):
    # Collect benign activations
    b_filebase = f"{RESULT_FOLDER}/b_"
    path = f"{TEMP_PATH}/benign"

    b_responses = sample_activations(b_samples, model, tokenizer, temp_dir=path, max_length=max_length)
    
    save_dict_to_json(b_responses, f"{b_filebase}generations.json")
    logger.info("Saved benign generations")
    
    benign_files = list(path.glob("*.pt"))
    b_activations = []
    if len(benign_files) != 0:
        for f in benign_files:
            b_activations.append(torch.load(f))
            
        b_activations = mean_activations_over_samples(b_activations)
        save_stack_to_pt(b_activations, f"{b_filebase}activations.pt")
        logger.info("Saved benign safe activations")
        
    del b_activations
    torch.cuda.empty_cache()

        
def h_sampling(h_samples: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int):
    # Collect harmful activations
    h_filebase = f"{RESULT_FOLDER}/h_"
    path = f"{TEMP_PATH}/harmful"

    h_responses = sample_activations(h_samples, model, tokenizer, temp_dir=path, max_length=max_length)

    save_dict_to_json(h_responses, f"{h_filebase}generations.json")
    logger.info("Saved harmful generations")
    
    harmful_files = list(path.glob("*.pt"))
    h_activations = []
    if len(harmful_files) != 0:
        for f in harmful_files:
            h_activations.append(torch.load(f))
            
        h_activations = mean_activations_over_samples(h_activations)
        save_stack_to_pt(h_activations, f"{h_filebase}activations.pt")
        logger.info("Saved harmful activations")
        
    del h_activations
    torch.cuda.empty_cache()
        
if __name__ == "__main__":
    logger.info("Starting experiment")
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Dataset path: %s", DATASET_PATH)
    model, tokenizer = MODEL()
    model.eval()
    
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
        
    samples, max_length = DATASET(tokenizer, B_ALL_KEY, B_CAT_KEY, H_ALL_KEY, H_CAT_KEY)

    b_sampling(samples[B_ALL_KEY], model, tokenizer, max_length)
    h_sampling(samples[H_ALL_KEY], model, tokenizer, max_length)
