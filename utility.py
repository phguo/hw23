# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023. All rights reserved.

import json


def load_json(file_name):
    with open(file_name, 'r') as f:
        lines = f.read()
        data = json.loads(
            lines, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    return data


def save_json(json_data, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(json_data, sort_keys=True, indent=4))
