import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

Q_TKN = "<usr>"
A_TKN = "<sys>"
SENT = "<unused1>"
EOS = "</s>"
BOS = "</s>"

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<unused0>",
)

model_path = "saved_models/chatbot_model.pth" 
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(
                koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)
            ).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(
                torch.argmax(pred, dim=-1).squeeze().numpy().tolist()
            )[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))