import json
import os
from collections import defaultdict
import gpt_wrapper
from gpt_wrapper.chat import Chat
from pylatexenc.latex2text import LatexNodes2Text
from openai import OpenAI

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
    
def save_to_jsonl(data, file_name):
    with open(file_name, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def prompt_for_qa(page):
    base_context = (
        "I am working on creating a supervised fine-tuning dataset based on lecture "
        "notes from university engineering courses. Just below is the content from a "
        "single page of lecture notes. I want you to create question/answer pairs based "
        "on insights gained from the content I've pasted below:\n\n"
        f"{page}\n\n"
    )
    specific_context = (
        "I chose to split my prompts by page from lecture notes, and as a result, some "
        "pages will be more informative than others. I therefore want you to generate "
        "between 1 and 3 question/answer pairs from the page of course notes I pasted just above, "
        "depending on the amount of useful/relevant content in the page. If there is no useful "
        "content in the page, generate a single generic question/answer pair related to the "
        "subject in the page (could be physics mechanics, quantum computing, natural language "
        "processing, machine learning, computer networks, cryptography, security and privacy, "
        "functional programming, etc...).\n\n"
    )
    more_specific_context = (
        "I want you to ensure that the question/answer pairs you provide are self-contained. This means the question "
        "should provide sufficient detail and context such that the accompanying answer could be obtained without "
        "relying on the lecture notes / page content I pasted above. In other words, any context from the lecture notes "
        "I pasted above should be provided in the question body if it is required to make the question answerable.\n\n"
    )
    even_more_specific_context = (
        "The question/answer pairs should be complex in order to serve as meaningful supervised fine-tuning examples. "
        "Generate difficult questions and ensure the corresponding answer is detailed and includes all relevant reasoning "
        "steps. It is far more important to have a few complex/in-depth than many short/shallow question/answer pairs.\n\n"
    )
    expected_response_structure_context = (
        "I want you to format your response exactly in the following manner:\n\n"
        "question: [question body ...]\n"
        "answer: [answer body ...]\n\n"
        "question: [question body ...]\n"
        "answer: [answer body ...]\n\n"
        "...\n\n"
        "Your response should match the above template exactly (the only difference being "
        "in the number of question/answer pairs you provide, i.e., between 1 and 3) and include "
        "no other accompanying description or context."
    )

    full_prompt = base_context + specific_context + more_specific_context + even_more_specific_context + expected_response_structure_context
    return full_prompt


def extract_qa_pairs(response):
    qa_pairs = []
    lines = response.strip().split('\n')
    question = None
    answer = None
    for line in lines:
        if line.startswith('question:'):
            if question and answer:
                qa_pairs.append({'question': question, 'chosen': answer})
            question = line[len('question: '):].strip()
            answer = None
        elif line.startswith('answer:'):
            answer = line[len('answer: '):].strip()
        else:
            if answer is not None:
                answer += ' ' + line.strip()
    if question and answer:
        qa_pairs.append({'question': question, 'chosen': answer})
    return qa_pairs


def call_openai_api(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def main(api_key, organisation, start_index=0):
    client = OpenAI(api_key=api_key, organization=organisation)

    course_dataset = load_json_file("course_material_latex.json")
    output_file = "qa_pairs.jsonl"

    for i, page in enumerate(course_dataset[start_index:], start=start_index):
        chat_prompt = prompt_for_qa(page)

        # CS-552 GPT wrapper call
        # chat = Chat.create("QA Chat")
        # response = chat.ask(chat_prompt)

        # OpenAI API call
        response = call_openai_api(client, chat_prompt)

        response = str(response)  # convert to str explicitely
        print(f"Processing page {i + 1}/{len(course_dataset)}")
        # print(response)

        qa_pairs = extract_qa_pairs(response)
        save_to_jsonl(qa_pairs, output_file)

        print(f"Saved {len(qa_pairs)} question/answer pairs to {output_file}")
        print("=" * 50)


if __name__ == "__main__":
    api_key = "..."
    organisation = "..."

    main(api_key, organisation)