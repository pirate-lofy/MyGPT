from configs import *
from BilingModelV2 import BigramLangModel


def get_batch(data):
    ix=torch.randint(len(data)-BLOCK,(BS,))
    x=torch.stack([data[i:i+BLOCK] for i in ix])
    y=torch.stack([data[i+1:i+BLOCK+1] for i in ix])
    return x.to(device),y.to(device)

@torch.no_grad()
def estimate_loss(model,data):
    losses=torch.zeros(EVAL_STEPS)
    for step in range(EVAL_STEPS):
        x,y=get_batch(data)
        _,loss=model(x,y)
        losses[step]=loss
    return losses.mean()


# load data
text=open('input.txt').read()
chars=sorted(list(set(text)))
n_chars=len(chars)

# encode - decode
ctoi={c:i for i,c in enumerate(chars)}
itoc={i:c for i,c in enumerate(chars)}
encode=lambda x:[ctoi[c] for c in x]
decode=lambda x:[itoc[i] for i in x]

# convert to tensor
data=torch.tensor(encode(text),dtype=torch.long)

# split
ratio=int(0.9*len(data))
trn_data=data[:ratio]
val_data=data[ratio:]


model=BigramLangModel(n_chars,N_EMBS,BLOCK).to(device)
# model=BigramLangModel(n_chars).to(device)
optim=torch.optim.AdamW(model.parameters(),LR)

print('before')
idx=model.generate(torch.zeros(1,1,dtype=torch.long).to(device),400)[0].tolist()
print(''.join(decode(idx)))

for epoch in range(EPOCHS):
    x,y=get_batch(trn_data)
    logits,loss=model(x,y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
print(loss)

print('\n\nafter')
idx=model.generate(torch.zeros(1,1,dtype=torch.long).to(device),400)[0].tolist()
print(''.join(decode(idx)))

