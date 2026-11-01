import json
for s in ("train", "val", "test"):
    d = json.load(open(f"dataset_v1/annotations/instances_{s}.json"))
    print(s, "| gambar:", len(d["images"]), "| kotak:", len(d["annotations"]),
          "| kategori:", [(c["id"], c["name"]) for c in d["categories"]])
