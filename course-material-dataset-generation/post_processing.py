import json
import os

keywords_for_filtering_out_qa = [
    "topic",
    "chapter",
    "authors",
    "authored",
    "author ",
    "lecture note",
    "course note",
    " notes",
    " course",
    "lecture",
]

phrases_to_remove = [
    # "as illustrated in the provided lecture notes",
    # "illustrated in the provided lecture notes",
    # "in the context of the lecture notes provided",
    # "in the provided lecture notes",
    # "according to the lecture notes",
    # "as mentioned in the lecture notes",
    # "as shown in the lecture notes",
    # "mentioned in the lecture notes",
    # "in the context discussed in the lecture notes", 
    # "in the context of the lecture notes",
    # "as discussed in the lecture notes",
    # "discussed in the lecture notes", 
    # "in the context provided in the lecture notes", 
    # "in the context of the scenario described in the lecture notes",
    # "as presented in the lecture notes",
    # "presented in the lecture notes",
    # "as described in the lecture notes",
    # "described in the lecture notes",
    # "as referenced in the lecture notes",
    # "referenced in the lecture notes",
    # "as given in the lecture notes",
    # "given in the lecture notes",
    # "as defined in the lecture notes",
    # "defined in the lecture notes",
    # "in the lecture notes"
]

def load_jsonl_file(file_name):
    data = []
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(data, file_name):
    with open(file_name, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def filter_out_keywords(qa_pairs):
    filtered_pairs = []
    for pair in qa_pairs:
        if not any(keyword in pair['question'].lower() or keyword in pair['chosen'].lower() for keyword in keywords_for_filtering_out_qa):
            filtered_pairs.append(pair)
    return filtered_pairs

def remove_phrases(qa_pairs):
    cleaned_pairs = []
    for pair in qa_pairs:
        question = pair['question']
        answer = pair['chosen']
        for phrase in phrases_to_remove:
            question = question.replace(phrase, "").strip()
            answer = answer.replace(phrase, "").strip()
        cleaned_pairs.append({'question': question, 'chosen': answer})
    return cleaned_pairs

def post_process_qa_pairs(input_file, output_file):
    qa_pairs = load_jsonl_file(input_file)
    qa_pairs = filter_out_keywords(qa_pairs)
    qa_pairs = remove_phrases(qa_pairs)
    save_to_jsonl(qa_pairs, output_file)
    print(f"Processed {len(qa_pairs)} Q/A pairs and saved to {output_file}")

if __name__ == "__main__":
    input_file = "qa_pairs.jsonl"
    output_file = "cleaned_qa_pairs.jsonl"
    post_process_qa_pairs(input_file, output_file)
