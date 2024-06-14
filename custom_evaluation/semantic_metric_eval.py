import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.model_selection import train_test_split
import evaluate
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from bart.bart_score import BARTScorer

torch.random.manual_seed(0)

# Paths
dataset_path = '../datasets/dpo_preference_example.jsonl'
local_model_path = "./local_models"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
model_name = "mlxha/mnlp-openaint-phi3-mini-dpo"

# Load the dataset
with open(dataset_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Split the dataset into train and test sets (use only test set for this example)
_, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to load or download model and tokenizer
def get_model_and_tokenizer(model_name: str, local_model_path):
    if os.path.exists(local_model_path):
        print("Loading model and tokenizer from local path...")
        model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="cuda", torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        print("Downloading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        os.makedirs(local_model_path, exist_ok=True)
        model.save_pretrained(local_model_path)
        tokenizer.save_pretrained(local_model_path)
        print(f"Model and tokenizer saved to {local_model_path}")
    return model, tokenizer


# Load or download the model and tokenizer
model, tokenizer = get_model_and_tokenizer(model_name, os.path.join(local_model_path, model_name))
tokenizer.padding_side  = 'left'  # address an error

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Format prompt to match expected phi-mini-3 template
batched_test_data = []
system_message = "Your are a helpful assistant specialized in technical course content. Provide detailed and accurate answers to students' questions."
for entry in test_data:
    formatted_prompt = pipe.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": entry['prompt']}
        ],
        tokenize=False
    )
    formatted_prompt = formatted_prompt + '<|assistant|>\n'
    batched_test_data.append(formatted_prompt)

# Predictions
outputs = pipe(batched_test_data, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.9, return_full_text=False, batch_size=16)

# Reformatting
predictions = []
for output in outputs:
    predictions.append(output[0]['generated_text'])



# COMPUTING METRICS -- EVALUATING PREDICTIONS

# references
chosen_answers = [entry['chosen'] for entry in test_data]

# Logging metrics
bleu_scores = []
bert_scores = []
bart_scores = []

# Metric objects
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart/bart_score.pth')

# BLEU evaluation
for idx, prediction in enumerate(predictions):
    reference = chosen_answers[idx]
    bleu_result = bleu.compute(predictions=[prediction], references=[[reference]])
    bleu_scores.append(bleu_result['bleu'])

# BERTscore evaluation
bert_results = bertscore.compute(predictions=predictions, references=chosen_answers, lang="en")
bert_scores = bert_results['f1']

# BARTscore evalutation
bart_scores = bart_scorer.score(srcs=predictions, tgts=chosen_answers, batch_size=4)

avg_bleu = np.mean(bleu_scores)
avg_bert = np.mean(bert_scores)
avg_bart = np.mean(bart_scores)
print(f"Average, BLEU: {avg_bleu:.3f}, BERTScore: {avg_bert:.3f}, BARTScore: {avg_bart:.3f}")


# Format logs
results = []
for idx in range(len(predictions)):
    result = {
        "prompt": test_data[idx]['prompt'],
        "prediction": predictions[idx],
        "reference": chosen_answers[idx],
        "bleu": bleu_scores[idx],
        "bertscore": bert_scores[idx],
        "bartscore": bart_scores[idx]
    }
    results.append(result)

# Save the results to a file
results_path = 'metric_logs.json'
with open(results_path, 'w') as file:
    json.dump(results, file, indent=4)

print(f"Results saved to {results_path}")

