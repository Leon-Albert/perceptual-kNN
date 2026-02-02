import torch
import tqdm

class graph_constructor: 

  def __init__(self,DF, max_iter_hubs, M_func):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.DF = DF.to(self.device)
    self.N = DF.size(dim=0)
    self.max_iter_hubs = max_iter_hubs
    self.M_func = M_func

  def initialize(self):
    #Initialize with 2 random hubs
    self.H = torch.randint(0,N,(2,1)) 
    self.M_hub = self.M_from_index(H)
    self.A = self.allocation()
    self.V = self.V_from_A()
    self.LP = [(H[0],H[1])]

  def M_from_index(self,Li):
    LM = []
    for i in Li:
      LM.append(self.M_func(self.DF[i,:]))
    return torch.stack(LM,dim=0).to(self.device)
  
  def V_from_A():
    V = [[] for l in self.H.size(dim=0)]
    for i in range(self.A.size(dim=0)):
      V[self.A[i]].append(i)
    return V
  
  def allocation(self):
    diff = self.DF[self.H].unsqueeze(0) - self.DF.unsqueeze(1)
    distances = torch.einsum('nka,kab,nkb->nk', diff, self.M_hub, diff)
    return torch.argmin(distances, dim=1)

  def criteria(self,iA,iB,iC):
    v_stack = torch.stack([
        self.DF[iA, :] - self.DF[iB, :],
        self.DF[iB, :] - self.DF[iC, :],
        self.DF[iA, :] - self.DF[iC, :],
        self.DF[iC, :] - self.DF[iA, :]
    ]).to(self.device)
    m_stack = torch.stack([
        self.M_hub[iB, :, :],
        self.M_hub[iB, :, :],
        self.M_hub[iA, :, :],
        self.M_hub[iC, :, :]
    ]).to(self.device)
    distances = torch.einsum('bn, bnm, bm -> b', v_stack, m_stack, v_stack)
    return (distances[0]+distances[1])/(distances[3]+distances[4])

  def choose_hub(self):
    Lkmax_pair = []
    LRk_pair = []
    for i_pair in range(len(self.LP)):
      pair = self.LP[i_pair]
      candidats = torch.cat((self.V[pair[0]][],self.V[pair[1]][]),0)
      LRk = []
      for k in candidats:
        Rk = 0
        for n in self.V[pair[0],:]:
          for m in self.V[pair[1],:]:
            Rk += self.criteria(n,k,m)
        LRk.append(Rk)
      Lkmax_pair.append(torch.argmax(LRk,dim=0))
      LRkmax_pair.append(LRk[Lkmax_pair[-1]])
    return Lkmax_pair[torch.argmax(LRkmax_pair,dim=0)]
  
  def new_hub(self):
    k = self.choose_hub()
    self.H = torch.cat((self.H, k), 0)
    self.M_hub = torch.cat((self.M_hub, self.M_from_index([k]).unsqeeze(0)), 0).to(self.device)
    self.A = self.allocation()
    self.V = self.V_from_A()

  def iter(self):
    old_A = self.A
    self.new_hub()
    LP = []
    for k in range(self.N):
      if old_A[k] != self.A[k]:
        LP.append((old_A[k],self.A[k]))
    self.LP = list(set(LP))

  def construct_hub(self):
    for i in tqdm.tqdm(range(self.max_iter_hubs),desc="Adding hubs...):
      iter(self)

  def construct_graph(self,k):
    return None
  
