from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import numpy as np
import pandas as pd

# Load the model, tokenizer and processor
model = AutoModelForCausalLM.from_pretrained("bytedance-research/ChatTS-14B", trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("bytedance-research/ChatTS-14B", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("bytedance-research/ChatTS-14B", trust_remote_code=True, tokenizer=tokenizer)

# Load your time series from CSV (no header)
df = pd.read_csv("/home/sdani/Smart_Food/simulated_data/test_outlier10.csv", sep=";", header=None)
df.columns = ['datetime', 'value']

# Use only a subset of the data for training the model
df = df.iloc[:120]
# Extract the values
#timeseries = df['value'].values.astype(np.float32)
timeseries = np.full(5, 2)
timeseries[2] = 7
#timeseries[13] = 7 
print(timeseries)


# Optional normalization (depending on model expectation)
# timeseries = (timeseries - np.mean(timeseries)) / np.std(timeseries)

# Create a prompt


prompt = f"I have a time series length of {len(timeseries)}: <ts><ts/>. Please tell me: 1. how long is the timeseries? 2. are there are any outliers or anomalies in this time series? 3. how many anomaly points are there? 4. in which exact time points do outliers or anomalies occur and with what values?"# Apply Chat Template
prompt = f"<|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user {prompt}<|im_end|><|im_start|>assistant"

# Convert to tensor
inputs = processor(text=[prompt], timeseries=[timeseries], padding=False, return_tensors="pt")

# Move tensors to the same device as model
for k in inputs:
    inputs[k] = inputs[k].to(model.device)

# Model Generate
outputs = model.generate(**inputs, max_new_tokens=300)

# Decode the output
print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
