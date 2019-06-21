import argparse
import json
import os.path
import uuid

def add_arguments(parser):
    parser.add_argument("--data_format", help="data format", required=True)
    parser.add_argument("--input_file", help="input data file", required=True)
    parser.add_argument("--output_file", help="output data file", required=True)

def preprocess(input_file,
               output_file,
               data_format):
    if not os.path.exists(input_file):
        raise FileNotFoundError("file not found")
    
    processed_data_list = []
    with open(input_file, "r") as file:
        token_list = []
        label_list = []
        for line in file:
            items = [item for item in line.strip().split(' ') if item]
            if len(items) == 0:
                if len(token_list) > 0 and len(label_list) > 0 and len(token_list) == len(label_list):
                    processed_data_list.append({
                        "id": str(uuid.uuid4()),
                        "text": " ".join(token_list),
                        "label": " ".join(label_list)
                    })
                
                token_list.clear()
                label_list.clear()
                continue
            
            if len(items) < 4:
                continue
            
            token = items[0]
            label = items[3]
            
            if token == "-DOCSTART-":
                continue
            
            token_list.append(token)
            label_list.append(label)
    
    if data_format == "json":
        save_json(processed_data_list, output_file)
    elif data_format == "text":
        save_text(processed_data_list, output_file)

def save_json(data_list,
              data_path):
    with open(data_path, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def save_text(data_list,
              data_path):
    with open(data_path, "wb") as file:
        for data in data_list:
            data_text = "{0}\t{1}\t{2}\r\n".format(data["id"], data["text"], data["label"])
            file.write(data_text.encode("utf-8"))

def main(args):
    preprocess(args.input_file, args.output_file, args.data_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
