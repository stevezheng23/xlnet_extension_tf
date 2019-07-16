import argparse
import json

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def convert_squad(input_file,
                  output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    output_dict = {}
    for data in input_data:
        qas_id = data["qas_id"]
        predict_text = data["predict_text"]
        
        output_dict[qas_id] = predict_text
    
    with open(output_file, "w") as file:
        json.dump(output_dict, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_squad(args.input_file, args.output_file)
