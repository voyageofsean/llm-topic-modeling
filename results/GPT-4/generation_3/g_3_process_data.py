"""
generation_3的思路：仿造generation_2进行，伪造数据，再讲输出改为三级。
用generation_2的生成结果生成符合generation_2输入格式的数据。
"""

import json

politics_data = []
output_data = []

with open("../generation_2.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = json.loads(line)
        if data["topics"][0][4:6] == "政治":
            politics_data.append(data)

for i, p_data in enumerate(politics_data):
    texts = p_data["text"].split("\n")
    topics = p_data["topics"][-1].split("\n")[1:]
    topics = list(map(lambda x: x.strip(), topics))
    for topic in topics:
        topic_label = topic.split(" ")[1]
        if "Document: " not in topic:
            continue
        topic_docs = topic.split("(")[1].split(")")[0].strip("Document: ")
        topic_docs = topic_docs.split(", ")
        start_num = int(texts[0].split(" ")[-1])
        for j in range(len(topic_docs)):
            output_data.append(
                {
                    "id": f"{i} Document {topic_docs[j]}",
                    "text": texts[(int(topic_docs[j]) - start_num) * 2 + 1],
                    "responses": "[1] " + topic_label + "：描述。"
                }
            )

with open("./g_3_pseudo_data.jsonl", 'w', encoding="utf-8") as f:
    for d in output_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')  # Ensure Chinese texts are written correctly

pass
