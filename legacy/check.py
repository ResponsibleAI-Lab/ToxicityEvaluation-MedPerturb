import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader
from tqdm import tqdm
from resources import TOKEN



class Testing:
    def __init__(self, model_path:str, 
                 tokenizer_path=None, 
                 top_p=1.0, 
                 temperature=1.0, 
                 top_k=50, 
                 multi_gpu=True, 
                 max_len=250,
                 batch_size=2,
                 input_file="data/sampled_dev.json", 
                 output_path="./predictions/") -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.multi_gpu = multi_gpu
        # self.model = self.initialize_model()  # Initialize the model during the class construction
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = max_len
        # self.UNCONDITIONAL_START_TOKEN = self.get_unconditional_start_token()
        # self.EOS_TOKEN_ID = self.tokenizer.eos_token_id
        # self.PAD_TOKEN_ID = self.tokenizer.eos_token_id 
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.input_file_abs_path = os.path.abspath(input_file)
        self.output_path = output_path
        # self.dataloader = dataloader.StereoSet(location=input_file)
        self.batch_size = batch_size




    def results_file_exists(self, clusters):
        """Check if the results file exists and contains the correct number of predictions."""
        # Extract model name
        model_name = os.path.basename(self.model_path)
        
        # Define the output filename based on model parameters
        output_filename = (
            f"{model_name}_temprature_{self.temperature}_nucleus_{self.top_p}_"
            f"maxlen_{self.max_len}_intrasentence_predictions.json"
        )
        
        output_file = os.path.join(self.output_path, output_filename)
        print(output_file)
        # Check if the output file exists
        if not os.path.exists(output_file):
            print("mistake")
            return False

        # Load predictions from the file
        with open(output_file, "r") as f:
            predictions = json.load(f)
        
        # Check if the number of predictions matches the number of clusters
        print(len(predictions))
        print(len(clusters))
        return len(predictions) == len(clusters)
    
dataloader = dataloader.StereoSet(location="data/sampled_dev.json")
stereoset_model = Testing(
                    model_path="microsoft/Phi-3-medium-128k-instruct",
                    top_p=0.85,
                    temperature=0.7,
                    multi_gpu=True,
                    max_len=300
                )

print(stereoset_model.results_file_exists(dataloader.get_intersentence_examples()))
    