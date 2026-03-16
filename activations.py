import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from pathlib import Path

from judge import SafetyJudge, RefusalJudge
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def all_sample_activations(samples, model, tokenizer, temp_dir, max_length=30):
    temp_dir  = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    responses = {}
    for i, sample in enumerate(samples):
        tokens = tokenizer(sample, return_tensors="pt", padding="max_length", max_length=max_length)
        tokens = tokens.to(model.device)
        output_input = model(**tokens, output_hidden_states=True)
        outputs = model.generate(
            **tokens,
            max_new_tokens=100,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            do_sample=False
        )
        generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        responses[sample] = generated

        activation_entry = tuple(h.cpu() for h in output_input.hidden_states)
        torch.save(activation_entry, temp_dir / f"sample_{i}.pt")

        logger.info("Finished sample %d out of %d", i + 1, len(samples))

        # free memory
        del outputs
        del activation_entry
        torch.cuda.empty_cache()
        
    logger.info("Completed extracting activations for %d samples", len(samples))
    return responses

def grouped_sample_activations(samples, model, tokenizer, safety_judge, refusal_judge, temp_dir, max_length=30):
    save_dir = Path(temp_dir)
    safe_dir = save_dir / "safe"
    unsafe_dir = save_dir / "unsafe"
    refusal_dir = save_dir / "refusal"

    # create folders
    safe_dir.mkdir(parents=True, exist_ok=True)
    unsafe_dir.mkdir(parents=True, exist_ok=True)
    refusal_dir.mkdir(parents=True, exist_ok=True)

    responses = []
    for i, sample in enumerate(samples):
        logger.info("[%d/%d] Sample: %s", i + 1, len(samples), sample)
        tokens = tokenizer(sample, return_tensors="pt", padding="max_length", max_length=max_length).to(model.device)
        output_input = model(**tokens, output_hidden_states=True)
        outputs = model.generate(
            **tokens,
            max_new_tokens=100,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            do_sample=False
        )

        generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        response = refusal_judge.judge_response(sample, generated)

        if response["judgement"] != "refusal":
            safety_response = safety_judge.judge_response(sample, generated)
            response["judgement"] = safety_response["judgement"]

        responses.append(response)

        logger.info("[%d/%d] Response: %s", i + 1, len(samples), response["judgement"])

        # move activations to CPU
        activation_entry = tuple(h.cpu() for h in output_input.hidden_states)

        # choose save path
        if response["judgement"] == "refusal":
            path = refusal_dir / f"sample_{i}.pt"
        elif response["judgement"] == "safe":
            path = safe_dir / f"sample_{i}.pt"
        else:
            path = unsafe_dir / f"sample_{i}.pt"

        torch.save(activation_entry, path)

        # free memory
        del outputs
        del activation_entry
        torch.cuda.empty_cache()

    logger.info("Grouped sampling completed")
    return responses


def mean_activations_over_samples(activations):
    num_layers = len(activations[0])
    sample_activations = []

    for layer_idx in range(num_layers):
        layer_values = []
        for sample in activations:
            h = sample[layer_idx][0, 1:, :]  # shape [seq_len, hidden_dim]
            token_mean = h.mean(dim=1)  # mean over seq_len
            layer_values.append(token_mean)
        layer_values = torch.stack(layer_values).mean(dim=0)
        sample_activations.append(layer_values)

    return torch.stack(sample_activations)


def sample_activations(samples, model, tokenizer, temp_dir=None, max_length=30, safety_judge=None, refusal_judge=None, grouped=False):
    if grouped:
        responses = grouped_sample_activations(samples, model, tokenizer, safety_judge, refusal_judge, temp_dir, max_length)
        return responses
    else:
        generated_responses = all_sample_activations(samples, model, tokenizer, temp_dir, max_length)
        return generated_responses