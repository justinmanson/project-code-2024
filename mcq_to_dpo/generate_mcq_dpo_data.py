import json
import os
from collections import defaultdict
import gpt_wrapper
from gpt_wrapper.chat import Chat
from pylatexenc.latex2text import LatexNodes2Text

# Set API base and key
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
# gpt_wrapper.api_key = "dd39a60d-11ad-4f8b-9057-53834d1a35ff"
gpt_wrapper.api_key = "4a44bf6b-e819-4313-b462-877e79c4d4ca"

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

def prompt_to_extract_correct_option(question, answer):
    prompt = (
        "Below is a multiple choice exam question and an open-worded response from a student. "
        "The open-worded response provides the students' correct chosen option from the choices along with detailed reasoning, "
        "however I am only interested in knowing which option the student has chosen as being correct. I therefore want you to "
        "analyse the worded response and extract only the letter corresponding to option chosen by the student. "
        "For instance, if within the open-worded response, the student writes: 'the correct answer is option B', then I want you to "
        "return only: 'B'.\n\n"
        f"Here is the question:\n {question}\n Here is the open-worded response from the student:\n {answer}\n"
        "Now tell me which letter the student chose as the correct answer/option. Remember your response should contain only a single letter!"
    )
    return prompt

def find_last_processed_index(output_file, input_data):
    if not os.path.exists(output_file):
        return 0

    with open(output_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return 0

        last_entry = json.loads(lines[-1])
        last_question = last_entry["prompt"].split("\nAnswer:")[0]

        for i, sample in enumerate(input_data):
            if sample["question_complete"] == last_question:
                return i + 1  # Start from the next entry

    return 0  # Start from the beginning if not found


def main(start_index=0):
    input_file = "M1_preference_data_15052024.json"
    output_file = "M1_mcq_dpo_data.jsonl"

    m1_preference_data = load_json_file(input_file)

    start_index = find_last_processed_index(output_file, m1_preference_data)

    for i, sample in enumerate(m1_preference_data[start_index:]):
        print(f"Processing sample {start_index+i+1} / {len(m1_preference_data)}.")

        dpo_entries = []  # list of {"prompt": ..., "chosen": ..., "rejected": ...} entries for each sample
        
        question = sample["question_complete"]

        preferences = sample["preference"]

        # print(f"Question body: {question}")
        
        chosens = []
        rejects = []
        # Iterate over preference pairs to get chosens
        for preference_pair in preferences:
            # chosen vs rejected id
            chosen_answer_id = preference_pair["overall"]
            rejected_answer_id = "A" if chosen_answer_id == "B" else "B"

            # Append to lists
            chosens.append(preference_pair[chosen_answer_id])
            rejects.append(preference_pair[rejected_answer_id])

        # If mcq type
        if "\n\nOptions:\n" in question:
            # Prompt chatgpt for correct option with every chosen
            for idx, chosen in enumerate(chosens):
                # print(f"Chosen response: {chosen}")

                chat = Chat.create("QA Chat")

                prompt = prompt_to_extract_correct_option(question, chosen)
                correct_option = chat.ask(prompt)
                correct_option = str(correct_option).strip()

                dpo_entries.append(
                    {
                        'prompt': f"{question}\nAnswer:",
                        'chosen': f"{correct_option}\n\n{chosen}",
                        'rejected': rejects[idx]
                    }
                )

        # If open question type
        else:
            for idx, chosen in enumerate(chosens):
                dpo_entries.append(
                    {
                        'prompt': question,
                        'chosen': chosen,
                        'rejected': rejects[idx]
                    }
                )

        # Dump samples to dpo dataset file
        save_to_jsonl(dpo_entries, output_file)



    print(f"MCQ-oriented DPO dataset saved to {output_file}")

if __name__ == "__main__":
    main()