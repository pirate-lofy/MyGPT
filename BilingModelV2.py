
from configs import *
from attention import Head


class BigramLangModel(nn.Module):
    def __init__(self,vocab_size,n_embs,block_size):
        super().__init__()
        # lookup table shows the relation of each charachter to all the other chars
        self.token_embedding_table=nn.Embedding(vocab_size,n_embs)
        self.pos_embedding_table=nn.Embedding(block_size,n_embs)
        self.sa_head=Head(n_embs,n_embs,BLOCK)
        self.lm_head=nn.Linear(n_embs,vocab_size)
    

    def forward(self,idx,targets=None):
        # idx and targets are both (B,T) 
        # T for timestep -> as the charachters are sequential (block_size)
        # C is the vocab_size, this dim is not naturally from data, got generated from 
        # the emb lookup table
        B,T=idx.shape
        tok_embs=self.token_embedding_table(idx) # (B,T,n_embs)
        pos_embs=self.pos_embedding_table(torch.arange(T,device=device)) # (T,n_emb)
        x=tok_embs+pos_embs # (B,T,n_embs)
        x=self.sa_head(x) # (B,T,n_emb)
        logits=self.lm_head(x) # (B,T,C) 

        if targets is None:
            return logits,None
        
        # compute loss
        B,T,C=logits.shape
        logits=logits.view(B*T,C) 
        targets=targets.view(B*T)
        loss=F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            logits,_=self(idx[:,-BLOCK:])
            logits=logits[:,-1,:] # (B,C) 
            prob=F.softmax(logits,dim=1) # (B,C) 
            idx_next=torch.multinomial(prob,num_samples=1) # (B,1) 
            idx=torch.cat([idx,idx_next],dim=1) # (B,T+1)
        return idx