import torch
import random
import os
import logging

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from models import *
from activations import *
from dataset_loaders import *
from visualisations import *
from utils import *

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
GROUPED = True
DATASET = load_wildjailbreak_dataset
MODEL = small_mistral
RESULT_FOLDER = f"results/{MODEL_PATH}/{DATASET_PATH}/{'grouped' if GROUPED else 'separated'}"
TEMP_PATH = f"temp_activations_{MODEL_PATH}_{DATASET_PATH}_{"grouped" if GROUPED else "separated"}"

B_ALL_KEY = "b_all"
B_CAT_KEY = "b_cat"
H_ALL_KEY = "h_all"
H_CAT_KEY = "h_cat"

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if api_key is None:
    raise ValueError("OpenRouter API key was not found")


def b_sampling(b_samples: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int, safety_judge: SafetyJudge, refusal_judge: RefusalJudge):
    b_filebase = f"{RESULT_FOLDER}/b_"

    logger.info("Starting benign sampling for %s samples", len(b_samples))

    b_responses = sample_activations(
        b_samples, model, tokenizer, 
        temp_dir=TEMP_PATH,
        max_length=max_length,
        safety_judge=safety_judge,
        refusal_judge=refusal_judge,
        grouped=True
    )

    b_responses = pd.DataFrame(b_responses)
    b_responses.to_csv(f"{b_filebase}generations.csv")
    logger.info("Saved benign generations")

    path = Path(TEMP_PATH) / "safe"
    safe_files = list(path.glob("*.pt"))
    b_safe = []

    if len(safe_files) > 0:
        for f in safe_files:
            b_safe.append(torch.load(f))

        # Compute mean across samples
        b_safe = mean_activations_over_samples(b_safe)

        # Save the mean activations
        save_stack_to_pt(b_safe, f"{b_filebase}safe_activations.pt")
        logger.info("Saved benign safe activations")
        
        del b_safe
        torch.cuda.empty_cache()

    path = Path(TEMP_PATH) / "unsafe"
    unsafe_files = list(path.glob("*.pt"))
    b_unsafe = []

    if len(unsafe_files) > 0:
        for f in unsafe_files:
            b_unsafe.append(torch.load(f))

        # Compute mean across samples
        b_unsafe = mean_activations_over_samples(b_unsafe)

        # Save the mean activations
        save_stack_to_pt(b_unsafe, f"{b_filebase}unsafe_activations.pt")
        logger.info("Saved benign unsafe activations")
        del b_unsafe
        torch.cuda.empty_cache()

    path = Path(TEMP_PATH) / "refusal"
    refusal_files = list(path.glob("*.pt"))
    b_refusal = []

    if len(refusal_files) > 0:
        for f in refusal_files:
            b_refusal.append(torch.load(f))

        # Compute mean across samples
        b_refusal = mean_activations_over_samples(b_refusal)

        # Save the mean activations
        save_stack_to_pt(b_refusal, f"{b_filebase}refusal_activations.pt")
        logger.info("Saved benign refusal activations")
        
        del b_refusal
        torch.cuda.empty_cache()

    os.rmdir(TEMP_PATH)


def h_sampling(h_samples: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int, safety_judge: SafetyJudge, refusal_judge: RefusalJudge):
    h_filebase = f"{RESULT_FOLDER}/h_"

    logger.info("Starting harmful sampling for %s samples", len(h_samples))

    h_responses = sample_activations(
        h_samples, model, tokenizer, 
        temp_dir=TEMP_PATH,
        max_length=max_length,
        safety_judge=safety_judge,
        refusal_judge=refusal_judge,
        grouped=True
    )

    h_responses = pd.DataFrame(h_responses)
    h_responses.to_csv(f"{h_filebase}generations.csv")
    logger.info("Saved harmful generations")

    path = Path(TEMP_PATH) / "safe"
    safe_files = list(path.glob("*.pt"))
    h_safe = []

    if len(safe_files) > 0:
        for f in safe_files:
            h_safe.append(torch.load(f))

        # Compute mean across samples
        h_safe = mean_activations_over_samples(h_safe)

        # Save the mean activations
        save_stack_to_pt(h_safe, f"{h_filebase}safe_activations.pt")
        logger.info("Saved harmful safe activations")
        
        del h_safe
        torch.cuda.empty_cache()

    path = Path(TEMP_PATH) / "unsafe"
    unsafe_files = list(path.glob("*.pt"))
    h_unsafe = []

    if len(unsafe_files) > 0:
        for f in unsafe_files:
            h_unsafe.append(torch.load(f))

        # Compute mean across samples
        h_unsafe = mean_activations_over_samples(h_unsafe)

        # Save the mean activations
        save_stack_to_pt(h_unsafe, f"{h_filebase}unsafe_activations.pt")
        logger.info("Saved harmful unsafe activations")
        del h_unsafe
        torch.cuda.empty_cache()

    path = Path(TEMP_PATH) / "refusal"
    refusal_files = list(path.glob("*.pt"))
    h_refusal = []

    if len(refusal_files) > 0:
        for f in refusal_files:
            h_refusal.append(torch.load(f))

        # Compute mean across samples
        h_refusal = mean_activations_over_samples(h_refusal)

        # Save the mean activations
        save_stack_to_pt(h_refusal, f"{h_filebase}refusal_activations.pt")
        logger.info("Saved harmful refusal activations")
        
        del h_refusal
        torch.cuda.empty_cache()
    
    os.rmdir(TEMP_PATH)

if __name__ == "__main__":
    logger.info("Starting experiment")
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Dataset path: %s", DATASET_PATH)
    
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
        logger.info("Created result folder at %s", RESULT_FOLDER)

    safety_judge = SafetyJudge(api_key)
    refusal_judge = RefusalJudge(api_key, model="meta-llama/llama-3-8b-instruct")

    model, tokenizer = MODEL()
    model.eval()

    logger.info("Model loaded")

    samples, max_length = DATASET(tokenizer, B_ALL_KEY, B_CAT_KEY, H_ALL_KEY, H_CAT_KEY)

    logger.info("Dataset loaded. Max prompt length: %s", max_length)
    
    if GROUPED:
        all_samples = list(samples[B_ALL_KEY]) + list(samples[H_ALL_KEY])
        h_sampling(all_samples, model, tokenizer, max_length, safety_judge, refusal_judge)
    else:
        b_sampling(samples[B_ALL_KEY], model, tokenizer, max_length, safety_judge, refusal_judge)
        h_sampling(samples[H_ALL_KEY], model, tokenizer, max_length, safety_judge, refusal_judge)

    logger.info("Experiment finished")