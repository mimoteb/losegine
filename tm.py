import os
import docx
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Seq2SeqTrainingArguments, Trainer

# Set working directory and model directory
WORKING_DIR = "/home/solomon/data/lose_data"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
DOCUMENTS_DIR = os.path.join(WORKING_DIR, "documents")

# Supported document types and their preprocessing functions
DOC_TYPES = {
    'pdf': (pdfminer.high_level.extract_text, 'rb'),
    'doc': (docx.Document, ''),
    'jpg': (pytesseract.image_to_string, 'rb'),
    'txt': (open, 'r')
}

class DocumentDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, doc_type = self.file_paths[idx]
        preprocess_func = DOC_TYPES.get(doc_type, (open, 'r'))
        with preprocess_func[0](file_path, preprocess_func[1]) as f:
            text = preprocess_func[0](f)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs

# Load files
file_paths = []
for root, dirs, files in os.walk(DOCUMENTS_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        doc_type = os.path.splitext(file)[1][1:]  # Extract file type
        if doc_type in DOC_TYPES:
            file_paths.append((file_path, doc_type))

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Create dataset and dataloader
dataset = DocumentDataset(file_paths, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=MODELS_DIR,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_steps=10,
    logging_dir=os.path.join(MODELS_DIR, "logs"),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained(MODELS_DIR)
tokenizer.save_pretrained(MODELS_DIR)