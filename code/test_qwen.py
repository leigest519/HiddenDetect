import torch
import torch.nn.functional as N
import csv
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
import random
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_datasets import *

refusal_lst = [
    "Sorry", "sorry",
    "unfortunately", "unfortunate", "sadly",
    "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful", "warning", "alarm", "caution",
    "shame", "conspiracy",
    "Subject", "contrary", "shouldn"
]

def make_context(
    tokenizer,
    query,
    history = None,
    system = "",
    max_window_size = 6144,
    chat_format = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return raw_text, context_tokens

def test(dataset, model_path, s=21, e=24):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    refusal_token_ids = []
    for token in refusal_lst:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        refusal_token_ids.append(token_id)
    token_one_hot = torch.zeros(vocab_size)
    for token_id in refusal_token_ids:
        token_one_hot[token_id] = 1.0

    lm_head = model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
    else:
        raise ValueError(f"Incorrect Model")

    label_all = []
    aware_auc_all = []

    for sample in dataset:
        F = []
        elements = []
        if sample["img"] is not None:
            elements.append({"image": sample["img"]})
        elements.append({"text": sample["txt"]})
        query = tokenizer.from_list_format(elements)
        raw_text, context_tokens = make_context(tokenizer=tokenizer, query=query)
        input_ids = torch.tensor([context_tokens]).to('cuda')

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        for i, r in enumerate(outputs.hidden_states[1:]):
            layer_output = norm(r)
            logits = lm_head(layer_output)
            next_token_logits = logits[:, -1, :]
            reference_tokens = token_one_hot.to(next_token_logits.device)
            cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
            F.append(cos_sim.item())

        F = F[s:e+1]
        if F:
            aware_auc = np.trapz(np.array(F))
        else:
            aware_auc = None

        label_all.append(sample['toxicity'])
        aware_auc_all.append(aware_auc)

    return label_all, aware_auc_all

def evaluate_AUPRC(true_labels, scores):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(true_labels, scores)
    auprc = auc(recall_arr, precision_arr)
    return auprc

def evaluate_AUROC(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    auroc = auc(fpr, tpr)
    return auroc

if __name__ == "__main__":
    model_path = "./model/Qwen-VL-Chat"
    datasets = {}
    results = {}

    datasets["XSTest"] = load_XSTest()
    datasets["FigTxt"] = load_FigTxt()
    datasets["MM-SafetyBench + MM-Vet"] = load_mm_safety_bench_all() + load_mm_vet()
    datasets["FigImg + MM-Vet"] = load_FigImg() + load_mm_vet()
    datasets["JBV28K_JBtxt + MM-Vet"] = load_JailBreakV_JBtxt() + load_mm_vet()
    datasets["JBV28K_JBtxt_SDimg + MM-Vet"] = load_JailBreakV_JBtxt_SDimg() + load_mm_vet()
    total_datasets = len(datasets)
    print(f"Starting evaluation of {total_datasets} datasets...")

    for idx, (dataset_name, dataset) in enumerate(datasets.items(), 1):
        print(f"Processing dataset {idx}/{total_datasets}: {dataset_name}")
        try:
            true_labels, scores = test(dataset, model_path, s=21, e=24)
            AUPRC = evaluate_AUPRC(true_labels, scores)
            AUROC = evaluate_AUROC(true_labels, scores)
            results[dataset_name] = (AUPRC, AUROC)
            print(f"AUPRC for {dataset_name}: {AUPRC}")
            print(f"AUROC for {dataset_name}: {AUROC}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    output_path = "result_qwen.csv"
    try:
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dataset Name", "AUPRC", "AUROC"])
            for dataset_name, result in results.items():
                if result is not None:
                    writer.writerow([dataset_name, f"{result[0]:.4f}", f"{result[1]:.4f}"])
        print(f"Results successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
