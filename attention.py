from configs import *

class Head(nn.Module):
    def __init__(self,n_emb,head_size,block_size):
        super().__init__()
        self.key=nn.Linear(n_emb,head_size,bias=False)
        self.query=nn.Linear(n_emb,head_size,bias=False)
        self.value=nn.Linear(n_emb,head_size,bias=False)
        # accumlative weighted sum
        # register to be saved as model weights
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape 
        # print('x',x.shape)
        # C is the head size = n_embs (sent as parameter)
        k=self.key(x) # (B,T,C)
        # print('k',k.shape)
        q=self.query(x) # (B,T,C)
        # print('q',q.shape)
        v=self.value(x) # (B,T,C)
        # print('v',v.shape)

        # computes attenstion scores (affinities)

        # in the paper "attention is all you need" it divides over sqrt(head size) dk for normalization
        # normalization is a must to prevent softmax from converging to a one-hot-encoded vector
        # as it will peek so much towards the biggest value
        
        # this step is just initializing the wei matrix
        wei=q@k.transpose(-2,-1)*C**-0.5 # (B,T,T)
        # print('w',wei.shape)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        # print(wei[0])
        wei=F.softmax(wei,dim=1)
        # print(wei[0])
        # print('w',wei.shape)
        out=wei@v # (B,T,T) @ (B,T,C) -> (B,T,C)
        # print('out',out.shape)
        return out
    

if __name__=='__main__':
    head=Head(N_EMBS,N_EMBS,BLOCK)
    x=torch.randn(4,8,32)
    head(x)