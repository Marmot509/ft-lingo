import json
import random

def formatter(input_file, output_file, start_index, end_index):
    # 读取 JSONL 文件并随机打乱
    with open(input_file, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]
        random.shuffle(jsonl_data)

    # 根据索引切割数据
    selected_data = jsonl_data[start_index:end_index]

    # 转换数据
    json_data = []
    for i, entry in enumerate(selected_data):
        conversation = [
            {
                "from": "human",
                "value": f"请续写歌词，第一句为：{entry['lyric'][0]}"
            },
            {
                "from": "model",
                "value": '，'.join(entry['lyric']) + '。'
            }
        ]
        json_data.append({"id": i + start_index, "conversations": conversation})

    # 将转换后的数据写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

# 输入和输出文件路径
source_data = 'lyric_data_for_CL_no_id.jsonl'

# 读取数据以计算数据集大小
with open(source_data, 'r', encoding='utf-8') as file:
    total_lines = sum(1 for line in file)

# 计算各数据集的索引范围
train_end_index = int(total_lines * 0.83)
val_start_index = train_end_index
val_end_index = val_start_index + int(total_lines * 0.12)
test_start_index = val_end_index

# 执行转换
formatter(source_data, 'train.json', 0, train_end_index)
formatter(source_data, 'train_mini.json', 0, 100)
formatter(source_data, 'val.json', val_start_index, val_end_index)
formatter(source_data, 'val_mini.json', val_start_index, val_start_index + 100)
formatter(source_data, 'test.json', test_start_index, total_lines)

