# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "../../../model-weights/gemma-2-27b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True, use_flash_attention_2=None
)
tokenizer.pad_token_id = tokenizer.eos_token_id
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

outputs = model.generate(input_ids=input_ids['input_ids'],
                         max_length = 600,
            attention_mask=input_ids['attention_mask'],
                         pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=32,  # Generate multiple completions per sentence
            do_sample=True,  # Enable sampling to generate diverse completions
            top_p=0.85,
            # top_k=self.top_k,
            temperature=0.7)
print(tokenizer.decode(outputs[0]))
