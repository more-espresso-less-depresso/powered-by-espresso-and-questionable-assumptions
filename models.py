from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration, MistralCommonBackend

def deepseek_qwen():
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def deepseek_llama():
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def deepseek_chat():
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def medium_mistral():
    model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Ministral-3-14B-Instruct-2512", device_map="auto")
    tokenizer = MistralCommonBackend.from_pretrained("mistralai/Ministral-3-14B-Instruct-2512")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def big_mistral():
    model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.2-24B-Instruct-2506", device_map="auto")
    tokenizer = MistralCommonBackend.from_pretrained("mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def small_mistral():
    model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512", device_map="auto")
    tokenizer = MistralCommonBackend.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def big_gemma():
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def small_gemma():
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def medium_gemma():
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer