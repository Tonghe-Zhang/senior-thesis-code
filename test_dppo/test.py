
import torch 
from collections import namedtuple
batch_size=1 #128
Batch = namedtuple("Batch", "actions conditions")


actions=torch.randn([batch_size,4,3])
conditions={'state':torch.randn([batch_size,1,11])}
batch = Batch(actions, conditions)

print(*batch)

