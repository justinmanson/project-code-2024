import json

def correct_key_in_jsonl(input_file, output_file):
    """
    Correct the key 'question' to 'prompt' in each JSON object in a JSONL file.

    Args:
    - input_file (str): Path to the input JSONL file.
    - output_file (str): Path to the output JSONL file with corrected keys.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            data['prompt'] = data.pop('question')
            data['chosen'] = data.pop('chosen')
            data['rejected'] = data.pop('rejected')
            outfile.write(json.dumps(data) + '\n')

# Example usage
input_file = "epfl_course_content_dpo_dataset.jsonl"
output_file = "corrected_epfl_course_content_dpo_dataset.jsonl"
correct_key_in_jsonl(input_file, output_file)