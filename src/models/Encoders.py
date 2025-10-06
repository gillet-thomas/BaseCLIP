import timm
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import (AutoModelForCausalLM, AutoTokenizer, DistilBertModel,
                          DistilBertTokenizer)


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Load model + Freeze all base model parameters
        self.model = timm.create_model(config["image_encoder"], pretrained=False, num_classes=0)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ImageEncoder2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create ResNet from scratch
        self.model = models.resnet50(weights=None)

        # Remove original classification layer
        # self.model.fc = nn.Identity()

        # Custom fine-tuning head
        # self.fine_tuning_head = nn.Sequential(
        #     nn.Linear(config["image_embedding"], config["image_embedding"]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(config["image_embedding"], config["image_embedding"])
        # )

        self.model.fc = nn.Linear(2048, config["image_embedding"])
        # nn.init.xavier_uniform_(self.model.fc.weight)
        # nn.init.zeros_(self.model.fc.bias)

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.model_name = config["text_encoder"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load model + Freeze all base model parameters
        self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Use the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0  ## Index 0 is CLS token represented by value 101

    def forward(self, text):
        tokenized_text = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

        # Move to device
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]  ## Output is shape (batch_size, hidden_size)


class TextSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, text):
        # summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        # print("Summary1", summarizer(sample_label, max_length=1000, min_length=30, do_sample=True))

        # summarizer = Summarizer()
        # print("SUMMARY2", summarizer(sample_label, num_sentences=5))

        # tokenizer = AutoTokenizer.from_pretrained("suriya7/bart-finetuned-text-summarization")
        # model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/bart-finetuned-text-summarization")
        # inputs = tokenizer([sample_label], max_length=1024, return_tensors='pt', truncation=True)
        # summary_ids = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
        # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print("SUMMARY3", summary)

        # generator = pipeline('text-generation', model='gpt2')
        # summary = generator("Summarize the following text: " + sample_label, max_length=130, num_return_sequences=1, do_sample=True)[0]['generated_text']
        # print("SUMMARY4", summary)

        prompt = "Provide a summary of this captions to only have one global caption: " + text
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpfuf text captions summarizer.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
