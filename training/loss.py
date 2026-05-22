import torch
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss,self).__init__()

        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def forward(self, embeddings, num_speakers, num_utterances):

        embeddings = embeddings.view(num_speakers, num_utterances, -1)

        centroids = torch.mean(embeddings, dim=1)

        embeddings_norm = F.normalize(embeddings, p=2, dim=2)
        centroids_norm = F.normalize(centroids, p=2, dim=1)

        sim_matrix = torch.einsum('nmd,cd->nmc', embeddings_norm, centroids_norm)
        sim_matrix = self.w* sim_matrix +self.b
        sim_matrix = sim_matrix.view(num_speakers * num_utterances, num_speakers)

        labels = torch.arange(num_speakers).repeat_interleave(num_utterances).to(embeddings.device)
        
        # Final loss calculation. Softmax applies under the hood here.
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    