import yaml
import json
p = "/home/dlrover/dlrover/python/tests/test.json"
def parse_yaml_file(file_path):
    data = None
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = file.read()
        data = yaml.safe_load(file_data)
    return data

data = parse_yaml_file("/home/dlrover/dlrover/python/tests/data/demo.yaml")
with open(p,'w') as f:
    json.dump(data, f)
