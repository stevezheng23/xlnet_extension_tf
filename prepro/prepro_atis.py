import argparse
import json
import os.path
import uuid

def add_arguments(parser):
    parser.add_argument("--input_dir", help="input data directory", required=True)
    parser.add_argument("--output_dir", help="output data directory", required=True)

def preprocess(input_dir,
               output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("directory not found")
    
    word_vocab_file = os.path.join(input_dir, "atis.dict.vocab.csv")
    word_vocab_list = read_text(word_vocab_file)
    
    intent_label_file = os.path.join(input_dir, "atis.dict.intent.csv")
    intent_label_list = read_text(intent_label_file)
    
    slot_label_file = os.path.join(input_dir, "atis.dict.slots.csv")
    slot_label_list = read_text(slot_label_file)
    
    train_query_file = os.path.join(input_dir, "atis.train.query.csv")
    train_query_list = read_text(train_query_file)
    
    train_intent_file = os.path.join(input_dir, "atis.train.intent.csv")
    train_intent_list = read_text(train_intent_file)
    
    train_slot_file = os.path.join(input_dir, "atis.train.slots.csv")
    train_slot_list = read_text(train_slot_file)
    
    train_raw_list = zip(train_query_list, train_intent_list, train_slot_list)
    train_processed_list = []
    for query_id, intent_id, slot_id in train_raw_list:
        train_data = {
            "id": str(uuid.uuid4()),
            "text": " ".join([word_vocab_list[int(token_vocab_id)] for token_vocab_id in query_id.split(' ')[1:-1]]),
            "token_label": " ".join([slot_label_list[int(token_slot_id)] for token_slot_id in slot_id.split(' ')[1:-1]]),
            "sent_label": intent_label_list[int(intent_id)],
        }
        
        train_processed_list.append(train_data)
    
    train_file = os.path.join(output_dir, "train-atis.json")
    save_json(train_processed_list, train_file)
    
    test_query_file = os.path.join(input_dir, "atis.test.query.csv")
    test_query_list = read_text(test_query_file)
    
    test_intent_file = os.path.join(input_dir, "atis.test.intent.csv")
    test_intent_list = read_text(test_intent_file)
    
    test_slot_file = os.path.join(input_dir, "atis.test.slots.csv")
    test_slot_list = read_text(test_slot_file)
    
    test_raw_list = zip(test_query_list, test_intent_list, test_slot_list)
    test_processed_list = []
    for query_id, intent_id, slot_id in test_raw_list:
        test_data = {
            "id": str(uuid.uuid4()),
            "text": " ".join([word_vocab_list[int(token_vocab_id)] for token_vocab_id in query_id.split(' ')[1:-1]]),
            "token_label": " ".join([slot_label_list[int(token_slot_id)] for token_slot_id in slot_id.split(' ')[1:-1]]),
            "sent_label": intent_label_list[int(intent_id)]
        }
        
        test_processed_list.append(test_data)
    
    test_file = os.path.join(output_dir, "test-atis.json")
    save_json(test_processed_list, test_file)

def read_text(data_path):
    if os.path.exists(data_path):
        with open(data_path, "r") as file:
            return [line.rstrip('\n') for line in file]
    else:
        raise FileNotFoundError("input file not found")

def read_json(data_path):
    if os.path.exists(data_path):
        with open(data_path, "r") as file:
            return json.load(file)
    else:
        raise FileNotFoundError("input file not found")

def save_text(data_list,
              data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "w") as file:
        for data in data_list:
            file.write("{0}\n".format(data))

def save_json(data_list,
              data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "w") as file:  
        json.dump(data_list, file, indent=4)

def main(args):
    preprocess(args.input_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
