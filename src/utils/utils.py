from typing import Any, List
import datetime
import pickle
import json
import yaml

def load_json_file(file_path: str) -> Any:
    with open(file_path, 'r') as file:
        return json.load(file)


def load_json_lines_file(file_path: str) -> Any:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_txt_file(file_path: str) -> List[str]:
    output = []
    with open(file_path, 'r') as file:
        for line in file:
            output.append(line.strip())
    return output


def load_pkl_file(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_yaml_file(file_path: str) -> Any:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_experiment_id():
    """ A function for generating unique experiment id based on the current time"""
    return (
        str(datetime.datetime.now())
        .replace(' ', '_')
        .replace(':', '_')
        .replace('.', '_')
    )
