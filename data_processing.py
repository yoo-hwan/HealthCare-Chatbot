import json
import csv

def preprocess_conversations(data):
    processed_data = []

    for item in data:
        conversation = item['conversations']
        for i in range(1, len(conversation), 2):
            q_id = item['id']
            q_text = conversation[i-1]['value']
            a_text = conversation[i]['value']

            processed_data.append({
                'id': q_id,
                'Q': q_text,
                'A': a_text
            })

    return processed_data

json_file_path = 'medical_conversation.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

processed_data = preprocess_conversations(json_data)

csv_file_path = 'medical_conversation.csv'
csv_columns = ['id', 'Q', 'A']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    for item in processed_data:
        writer.writerow(item)

print(f"Data has been processed and saved to {csv_file_path}")
