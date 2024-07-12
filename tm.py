import os
import glob
import pdfminer
import python_docx
import pytesseract
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Document preprocessing
def preprocess_documents(doc_type, file_path):
    if doc_type == 'pdf':
        # Use pdfminer to extract text from PDF file
        with open(file_path, 'rb') as f:
            text = pdfminer.extractText(f)
    elif doc_type == 'doc':
        # Use python-docx to extract text from DOC file
        doc = python_docx.Document(file_path)
        text = ''.join([p.text for p in doc.paragraphs])
    elif doc_type == 'jpg':
        # Use pytesseract to extract text from JPG file
        text = pytesseract.image_to_string(file_path)
    elif doc_type == 'txt':
        # Simply read the text from TXT file
        with open(file_path, 'r') as f:
            text = f.read()
    return text

# Create dataset class
class DocumentDataset:
    def __init__(self, files, tokenizer):
        self.files = files
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        file_path = self.files[idx]
        doc_type = file_path.split('.')[-1]
        text = preprocess_documents(doc_type, file_path)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs

    def __len__(self):
        return len(self.files)

# Create dataset and data loader
files = glob.glob('path/to/your/documents/*')
tokenizer = AutoTokenizer.from_pretrained('your_pretrained_model')
dataset = DocumentDataset(files, tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Fine-tune the model
model = AutoModelForQuestionAnswering.from_pretrained('your_pretrained_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
    model.eval()