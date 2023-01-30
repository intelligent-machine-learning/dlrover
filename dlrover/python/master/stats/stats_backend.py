
import os
import yaml
import json


def parse_json_file(file_path):
    data = None
    with open(file_path,"r") as f:
        data = json.load(f)
    return data

def parse_yaml_file(file_path):
    data = None
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = file.read()
        data = yaml.safe_load(file_data)
    return data

class LocalFileStateBackend:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {} 

    def load(self):
        data = {}
        if self.file_path.endswith("json"):
            data = parse_json_file(self.file_path)
        elif self.file_path.endswith("yaml"):
            data =parse_yaml_file(self.file_path)
        else:
            print("error") # to do: logging 
        self.data = data
        return data

    def get(self, key, default_value=None):
        return self.data.get(key, default_value)

    def put(self, key, value):
        self.data[key] = value