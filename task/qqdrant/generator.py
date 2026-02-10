from faker import Faker
import json

fake = Faker()

docs = []
for i in range(200):
    docs.append({
        "id": i,
        "title": fake.sentence(),
        "content": fake.text(max_nb_chars=1000)
    })

with open("test_docs.json", "w") as f:
    json.dump(docs, f, indent=2)

print("Documents generated!")
