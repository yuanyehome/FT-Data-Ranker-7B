import argparse
import transformers

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("output")
args = parser.parse_args()

model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
)
model = model.half()
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
