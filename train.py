import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re, os
from tqdm import tqdm


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)


print("start1")


class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  
        return len(self._data)

    def __getitem__(self, idx):  
        turn = self._data.iloc[idx]
        q = turn["Q"]  
        q = re.sub(r"([?.!,])", r" ", q)  

        a = turn["A"]  
        a = re.sub(r"([?.!,])", r" ", a)  

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        
        if q_len > self.max_len:
            a_len = self.max_len - q_len  
            if a_len <= 0:  
                q_toked = q_toked[-(int(self.max_len / 2)) :]  
                q_len = len(q_toked)
                a_len = self.max_len - q_len  
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len  
            if a_len <= 0:  
                q_toked = q_toked[-(int(self.max_len / 2)) :]  
                q_len = len(q_toked)
                a_len = self.max_len - q_len  
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]

        
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        
        return (token_ids, np.array(mask), labels_ids)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token="<unk>",
    pad_token=PAD,
    mask_token=MASK,
)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

dataname = "medical_conversation.csv"
Chatbot_Data = pd.read_csv("./"+ dataname)
Chatbot_Data.dropna(subset=["A"], inplace=True)
Chatbot_Data.dropna(subset=["Q"], inplace=True)
Chatbot_Data.dropna(subset=["id"], inplace=True)


print("start3")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ChatbotDataset(Chatbot_Data, max_len=40)
train_dataloader = DataLoader(
    train_set,
    batch_size=32,
    num_workers=0,
    shuffle=True,
    collate_fn=collate_batch,
)

model.to(device)

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 10
Sneg = -1e18


for epoch in range(epoch):
    dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch_idx, samples in enumerate(dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        token_ids, mask, label = token_ids.to(device), mask.to(device), label.to(device)
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()


model_save_path = os.path.join(save_dir, "chatbot_model.pth")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    },
    model_save_path,
)

print("Model saved at:", model_save_path)