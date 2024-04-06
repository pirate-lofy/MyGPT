
from configs import *


class BigramLangModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # lookup table shows the relation of each charachter to all the other chars
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        # idx and targets are both (B,T) 
        # T for timestep -> as the charachters are sequential
        logits=self.token_embedding_table(idx) # (B,T,C)
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
            logits,_=self(idx) 
            logits=logits[:,-1,:] # (B,C) 
            prob=F.softmax(logits,dim=1) # (B,C) 
            idx_next=torch.multinomial(prob,num_samples=1) # (B,1) 
            idx=torch.cat([idx,idx_next],dim=1) # (B,T+1)
        return idx