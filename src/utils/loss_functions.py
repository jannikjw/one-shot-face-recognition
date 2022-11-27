import time
# from sqlalchemy import func
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def cal_distance(self, x1: torch.Tensor, x2: torch.Tensor, func = None):
        if func == None:
            # if there is no function is given then use the euclidean distance function
            func = self.euclidean_dist
        
        return func(x1, x2)

    def euclidean_dist(self, x1: torch.Tensor, x2: torch.Tensor):
        return (x1-x2).pow(2).sum(axis=1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = self.cal_distance(anchor, positive)
        neg_dist = self.cal_distance(anchor, negative)
        # print("Pos loss is: {}, neg loss is {}".format(pos_dist, neg_dist))
        
        losses =  torch.relu(pos_dist - neg_dist + self.margin)

        return losses.mean()
