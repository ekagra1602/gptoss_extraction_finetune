from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",    # lets HF pick best dtype; float16 for MPS likely
    device_map="auto",
)

outputs = pipe(
    [{"role": "user", "content": "Explain quantum mechanics concisely."}],
    max_new_tokens=128,
)

print(outputs[0]["generated_text"])
