import os
import pdfminer.high_level
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Set working directory and model directory
WORKING_DIR = "/home/solomon/data/lose_data"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
DOCUMENTS_DIR = os.path.join(WORKING_DIR, "documents")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Supported document type and its preprocessing function
DOC_TYPE = {
    'txt': (open, 'r'),
    # Add more file types here if necessary
}

import chardet

class DocumentDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, doc_type = self.file_paths[idx]
        preprocess_func = DOC_TYPE.get(doc_type, (open, 'r'))

        with preprocess_func[0](file_path, preprocess_func[1]) as f:
            text = f.read()

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Ensure correct dimensionality
        if inputs['input_ids'].ndim > 2:
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        return inputs



# Load only supported files
file_paths = []
for root, dirs, files in os.walk(DOCUMENTS_DIR):
    for file in files:
        if (doc_type := os.path.splitext(file)[1][1:]) in DOC_TYPE:
            file_paths.append((os.path.join(root, file), doc_type))

if not file_paths:
    raise ValueError("No supported document files found.")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForQuestionAnswering.from_pretrained('xlm-roberta-base')

# Create dataset and dataloader
dataset = DocumentDataset(file_paths, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

# Training arguments
training_args = TrainingArguments(
    output_dir=MODELS_DIR,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    logging_dir=os.path.join(MODELS_DIR, "logs"),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train and save
trainer.train()
model.save_pretrained(MODELS_DIR)
tokenizer.save_pretrained(MODELS_DIR)
