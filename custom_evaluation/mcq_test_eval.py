import json
import os
import gpt_wrapper
from gpt_wrapper.chat import Chat
from collections import defaultdict

# Set API base and key
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "4a44bf6b-e819-4313-b462-877e79c4d4ca"

# Constant to determine whether to generate answers
GENERATE_ANSWERS = False

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

def prompt_for_answer(question):
    prompt = (
        f"Answer the following multiple-choice question with a single letter (A, B, C, or D):\n\n"
        f"{question}\n"
        "Answer: "
    )
    return prompt

def calculate_accuracy(results):
    correct_predictions = sum(1 for result in results if result['predicted_answer'] == result['real_answer'])
    global_accuracy = correct_predictions / len(results) * 100
    
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    
    for result in results:
        subject = result['subject']
        subject_total[subject] += 1
        if result['predicted_answer'] == result['real_answer']:
            subject_correct[subject] += 1
    
    subject_accuracy = {subject: (subject_correct[subject] / subject_total[subject]) * 100 for subject in subject_total}
    
    return global_accuracy, subject_accuracy

def generate_and_evaluate_answers(test_data, output_file):
    results = []
    for i, entry in enumerate(test_data):
        print(f"Processing question {i+1} of {len(test_data)}")
        chat = Chat.create("QA Chat")
        question_prompt = prompt_for_answer(entry['question'])
        response = chat.ask(question_prompt)

        # Split on the full stop and take the first element
        predicted_answer = str(response).strip().split('.')[0]

        result = {
            "subject": entry["subject"],
            "question": entry["question"],
            "real_answer": entry["answer"],
            "predicted_answer": predicted_answer
        }
        results.append(result)
    
    save_to_jsonl(results, output_file)
    return results

def main():
    input_file = "all-exams-testing-set.jsonl"
    output_file = "gpt_3dot5_test_results.jsonl"

    test_data = load_jsonl_file(input_file)
    
    if GENERATE_ANSWERS:
        # Generate answers and evaluate
        results = generate_and_evaluate_answers(test_data, output_file)
    else:
        # Load the results directly from the output file
        results = load_jsonl_file(output_file)

    # Calculate accuracy
    global_accuracy, subject_accuracy = calculate_accuracy(results)
    print(f"Global accuracy over the test set: {global_accuracy:.2f}%")
    for subject, accuracy in subject_accuracy.items():
        print(f"Accuracy for {subject}: {accuracy:.2f}%")

    print(f"Test results saved to {output_file}")

if __name__ == "__main__":
    main()
