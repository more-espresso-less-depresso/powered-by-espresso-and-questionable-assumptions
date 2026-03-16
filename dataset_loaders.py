import os
import random
import logging

from dotenv import load_dotenv
from datasets import load_dataset

from judge import DatasetJudge
from utils import save_list_to_jsonl

random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DATASET_FOLDER = "datasets"
NO_SAMPLES = 550

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if api_key is None:
    raise ValueError("OpenRouter API key was not found")

def get_max_length(tokenizer, samples, samples2=None):
    max_length_samples = max([len(tokenizer(sample)["input_ids"]) for sample in samples])
    if samples2 is not None:
        max_length_samples2 = max([len(tokenizer(sample)["input_ids"]) for sample in samples2])
        return max(max_length_samples, max_length_samples2) + 1
    return max_length_samples + 1

def generate_benign_samples(all_harmful_samples, prompt_column) -> list[str]:
    dataset_judge = DatasetJudge(api_key, model="openai/gpt-5.4")
    idxs_all_samples = set(range(len(all_harmful_samples)))
    all_benign_samples = []
    new_harmful_samples = []
    current_idxs = set()

    while len(all_benign_samples) < NO_SAMPLES:
        new_idx = random.choice(list(idxs_all_samples - current_idxs))
        new_benign = dataset_judge.judge_response(all_harmful_samples[new_idx])
        if new_benign is not None:
            all_benign_samples.append({prompt_column: new_benign["response"]})
            new_harmful_samples.append({prompt_column: all_harmful_samples[new_idx]})
        else:
            logger.info("Failed to generate benign sample for index %s, current number of benign samples: %s", new_idx, len(all_benign_samples))
        current_idxs.add(new_idx)

    return all_benign_samples, new_harmful_samples

def load_harmdirect_dataset(tokenizer, b_all_key, b_cat_key, h_all_key, h_cat_key) -> tuple[dict, int]:
    prompt_column = "Question"
    harmful_set = load_dataset("SoftMINER-Group/HarmEval", split="train")

    all_harmful_samples = harmful_set[prompt_column]
    harmful_samples = {
        category: harmful_set.filter(lambda x: x["Topic"] == category)[prompt_column]
        for category in set(harmful_set["Topic"])
    }

    benign_path = f"{DATASET_FOLDER}/harm_direct_benign.jsonl"

    if os.path.exists(benign_path):
        benign_set = load_dataset("json", data_files=benign_path, split="train")
        all_benign_samples = benign_set[prompt_column]
        benign_samples = {
            category: benign_set.filter(lambda x: x["Topic"] == category)[prompt_column]
            for category in set(benign_set["Topic"])
        }
    else:
        benign_samples = {
            category: generate_benign_samples(harmful_samples[category], prompt_column)
            for category in harmful_samples
        }

        all_benign_samples = [
            {prompt_column: sample, "Topic": category}
            for category, samples in benign_samples.items()
            for sample in samples
        ]

        save_list_to_jsonl(all_benign_samples, benign_path)
        all_benign_samples = [item[prompt_column] for item in all_benign_samples]

    if tokenizer:
        max_length_harmful = get_max_length(tokenizer, all_benign_samples, all_harmful_samples)
    else:
        max_length_harmful = 1

    logger.info("Harmful set size: %s", len(harmful_set))
    logger.info("Benign set size: %s", len(benign_set))
    logger.info("The longest prompt is %s", max_length_harmful)

    samples = {
        b_all_key: all_benign_samples,
        b_cat_key: None,
        h_all_key: all_harmful_samples,
        h_cat_key: None
    }

    return samples, max_length_harmful


def load_harmcontext_dataset(tokenizer, b_all_key, b_cat_key, h_all_key, h_cat_key) -> tuple[dict, int]:
    prompt_column = "adversarial"

    full_eval_set = load_dataset(
        "allenai/wildjailbreak",
        "eval",
        split="train",
        delimiter="\t",
        keep_default_na=False
    )

    harmful_set = full_eval_set.filter(lambda x: x["data_type"] in ["adversarial_harmful"])
    all_harmful_samples = harmful_set[prompt_column]

    benign_path = f"{DATASET_FOLDER}/harm_context_benign.jsonl"
    harmful_path = f"{DATASET_FOLDER}/harm_context_harmful.jsonl"

    if os.path.exists(benign_path):
        benign_set = load_dataset("json", data_files=benign_path, split="train")
        all_benign_samples = benign_set[prompt_column]

        harmful_set = load_dataset("json", data_files=harmful_path, split="train")
        all_harmful_samples = harmful_set[prompt_column]
    else:
        all_benign_samples, all_harmful_samples = generate_benign_samples(all_harmful_samples, prompt_column)
        save_list_to_jsonl(all_benign_samples, benign_path)
        save_list_to_jsonl(all_harmful_samples, harmful_path)
        all_benign_samples = [item[prompt_column] for item in all_benign_samples]
        all_harmful_samples = [item[prompt_column] for item in all_harmful_samples]

    if tokenizer:
        max_length_harmful = get_max_length(tokenizer, all_benign_samples, all_harmful_samples)
    else:
        max_length_harmful = 1

    logger.info("Harmful set size: %s", len(all_harmful_samples))
    logger.info("Benign set size: %s", len(all_benign_samples))
    logger.info("The longest prompt is %s", max_length_harmful)

    samples = {
        b_all_key: all_benign_samples,
        b_cat_key: None,
        h_all_key: all_harmful_samples,
        h_cat_key: None
    }

    return samples, max_length_harmful