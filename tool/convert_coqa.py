import argparse
import json

import numpy as np

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--answer_threshold", help="threshold of answer", required=False, default=0.1, type=float)

def convert_coqa(input_file,
                 output_file,
                 answer_threshold):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    output_data = []
    for data in input_data:
        id_items = data["qas_id"].split('_')
        id = id_items[0]
        turn_id = int(id_items[1])
        
        prob_list = [data["unk_prob"], data["yes_prob"], data["no_prob"]]
        answer_list = ["unknown", "yes", "no"]
        
        prob_idx = np.argmax(prob_list)
        if prob_list[prob_idx] >= answer_threshold:
            answer = answer_list[prob_idx]
            if answer == "yes" and "true or false" in data["question_text"].lower():
                answer = "true"
            elif answer == "no" and "true or false" in data["question_text"].lower():
                answer = "false"
        else:
            answer = data["predict_text"]
        
        score = prob_list[prob_idx]
        
        output_data.append({
            "id": id,
            "turn_id": turn_id,
            "answer": answer,
            "score": score
        })
    
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_coqa(args.input_file, args.output_file, args.answer_threshold)
