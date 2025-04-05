import torch
from transformers import BartForConditionalGeneration, BartTokenizer

class TransformerAutoEncoder:
    def __init__(self, model_name="facebook/bart-base", device="cpu", max_length=128):
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.max_length = max_length

    def tokenize_prompts(self, prompts):
        # Tokenize the raw text into input_ids, attention_mask, etc.
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        return inputs

    def forward(self, prompts):
        inputs = self.tokenize_prompts(prompts)
        # Here we use input_ids as labels so that the model learns to reconstruct
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        return loss, outputs.logits

    def reconstruct(self, prompt):
        # Generate a reconstruction for a single prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        output_ids = self.model.generate(**inputs, max_length=self.max_length)
        reconstructed_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return reconstructed_text
