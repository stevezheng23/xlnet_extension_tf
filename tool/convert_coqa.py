import argparse
import json

import numpy as np

from eval_coqa import CoQAEvaluator

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
        
        score_list = [data["unk_score"], data["yes_score"], data["no_score"], data["num_score"], data["opt_score"]]
        answer_list = ["unknown", "yes", "no", "number", "option"]
        
        score_idx = np.argmax(score_list)
        if score_list[score_idx] >= answer_threshold:
            answer = answer_list[score_idx]
            if answer == "number":
                answer_list = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
                answer = answer_list[data["num_id"]-1]
            elif answer == "option":
                answer = data["predict_text"]
                norm_question_tokens = CoQAEvaluator.normalize_answer(data["question_text"]).split(" ")
                if "or" in norm_question_tokens:
                    index = norm_question_tokens.index("or")
                    if index-1 >= 0 and index+1 < len(norm_question_tokens):
                        answer_list = [norm_question_tokens[index-1], norm_question_tokens[index+1]]
                        answer = answer_list[data["opt_id"]-1]
        else:
            answer = data["predict_text"]
        
        output_data.append({
            "id": id,
            "turn_id": turn_id,
            "answer": answer
        })
    
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_coqa(args.input_file, args.output_file, args.answer_threshold)
