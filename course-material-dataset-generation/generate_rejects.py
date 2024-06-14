import json
import os
from collections import defaultdict
import gpt_wrapper
from gpt_wrapper.chat import Chat
from pylatexenc.latex2text import LatexNodes2Text

# Set API base and key
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "..."

def render_latex_to_text(latex_str):
    return LatexNodes2Text().latex_to_text(latex_str)

def load_json_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            return json.load(file)
    else:
        # If the file doesn't exist, create it with an empty list
        with open(file_name, 'w') as file:
            json.dump([], file)
        return []

def load_jsonl_file(file_name):
    data = []
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(data, file_name):
    with open(file_name, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def prompt_for_low_quality_answer(question):
    # prompt = (
    #     "I want you to generate a low-quality, at least partially incorrect, and less informative answer "
    #     "to the following question. The answer should contain inaccuracies and lack details, "
    #     "though not be completely gibberish. Here is the question:\n\n"
    #     f"question: {question}\n"
    #     "answer: "
    # )
    prompt = (
        f"Answer the following question:\n\n {question}\n"
    )
    return prompt

def generate_and_save_rejects(qa_pairs, file_name):
    for pair in qa_pairs:
        chat = Chat.create("QA Chat")
        low_quality_prompt = prompt_for_low_quality_answer(pair['question'])
        response = chat.ask(low_quality_prompt)
        pair['rejected'] = str(response).strip()
        save_to_jsonl([pair], file_name)

def main(start_index=0):
    input_file = "cleaned_qa_pairs.jsonl"
    output_file = "epfl_course_content_dpo_dataset.jsonl"

    qa_pairs = load_jsonl_file(input_file)

    # Generate low-quality answers
    generate_and_save_rejects(qa_pairs, output_file)

    print(f"Final dataset with low-quality responses saved to {output_file}")

if __name__ == "__main__":
    main()