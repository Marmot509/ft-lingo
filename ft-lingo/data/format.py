import json

def formatter(input_file, output_file):
    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]

    # 转换数据
    json_data = []
    for i, entry in enumerate(jsonl_data):
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
        json_data.append({"id": i, "conversations": conversation})
        if i > 100:
            break

    # 将转换后的数据写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

# 输入和输出文件路径
input_file = 'lyric_data_for_CL_no_id.jsonl'
output_file = 'lyrics.json'

# 执行转换
formatter(input_file, output_file)
