import argparse
import random
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(args.encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def generate(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_len
        )
        with torch.inference_mode():
            outs = self.llm.generate(**inputs, max_new_tokens=8)
        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)

def classify_output(output, choices):
    out = output.lower().strip()
    if "sol1" in out or out.startswith("a"): return "A"
    if "sol2" in out or out.startswith("b"): return "B"
    return "?"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="piqa")   # giữ để script không lỗi
    parser.add_argument("--encoder", default="google/flan-t5-small")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    set_random_seed(args.seed)

    model = MyModel(args)

    # Load PIQA dataset
    ds = load_dataset("piqa")
    val = ds["validation"]

    correct_count = 0
    total = 5   # chỉ chạy 5 câu

    for i in range(total):
        ex = val[i]
        q = ex["goal"]
        choices = [ex["sol1"], ex["sol2"]]
        correct = "A" if ex["label"] == 0 else "B"

        text = f"Question: {q}\nOption A: {choices[0]}\nOption B: {choices[1]}\nAnswer:"
        output = model.generate([text])[0]
        pred = classify_output(output, choices)

        print("===================================")
        print(f"Question {i+1}: {q}")
        print(f"Option A: {choices[0]}")
        print(f"Option B: {choices[1]}")
        print(f"Model Output: {output}")
        print(f"Model Chose: {pred}")
        print(f"Correct Answer: {correct}")
        if pred == correct:
            print("✅ Correct")
            correct_count += 1
        elif pred == "?":
            print("❓ Cannot classify")
        else:
            print("❌ Incorrect")

    print("===================================")
    print(f"Accuracy: {correct_count}/{total} = {round(correct_count/total*100,1)}%")

if __name__ == "__main__":
    main()
