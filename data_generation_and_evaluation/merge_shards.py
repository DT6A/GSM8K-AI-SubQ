import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_count", type=int, default=74)
    parser.add_argument("--load_dir", type=str, default="chat_data")
    parser.add_argument("--save_path", type=str, default="chat_data/merged_data.jsonl")
    parser.add_argument("--no_feedback", action="store_true")
    args = parser.parse_args()

    data = []
    sample_idx = 0
    for i in range(args.shards_count + 1):
        if not args.no_feedback:
            with open(f"{args.load_dir}/generated_feedback_{i}.jsonl", "r") as f_file:
                with open(f"{args.load_dir}/generated_answers_{i}.jsonl", "r") as a_file:
                    for fline, aline in zip(f_file, a_file):
                        fdata_dict = json.loads(fline)
                        adata_dict = json.loads(aline)
                        fdata_dict["answers"] = adata_dict["answers"]
                        fdata_dict["answer"] = adata_dict["answer"]
                        data.append(fdata_dict)
        else:
            with open(f"{args.load_dir}/generated_answers_{i}.jsonl", "r") as a_file:
                for aline in a_file:
                    adata_dict = json.loads(aline)
                    data.append(adata_dict)
    with open(args.save_name, "w") as json_file:
        for d in data:
            json_file.write(json.dumps(d) + '\n')
