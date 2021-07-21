import torch
from torch import nn

class EarlyRewardLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=10):
        super(EarlyRewardLoss, self).__init__()

        self.negative_log_likelihood = nn.NLLLoss(reduction="none")
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, log_class_probabilities, probability_stopping, y_true, return_stats=False):
        N, T, C = log_class_probabilities.shape

        # equation 3
        Pt = calculate_probability_making_decision(probability_stopping)

        # equation 7 additive smoothing
        Pt = Pt + self.epsilon / T

        # equation 6, right term
        t = torch.ones(N, T, device=log_class_probabilities.device) * \
                  torch.arange(T).type(torch.FloatTensor).to(log_class_probabilities.device)

        earliness_reward = Pt * probability_correct_class(log_class_probabilities, y_true) * (1 - t / T)
        earliness_reward = earliness_reward.sum(1).mean(0)

        # equation 6 left term
        cross_entropy = self.negative_log_likelihood(log_class_probabilities.view(N*T,C), y_true.view(N*T)).view(N,T)
        classification_loss = (cross_entropy * Pt).sum(1).mean(0)

        # equation 6
        loss = self.alpha * classification_loss - (1-self.alpha) * earliness_reward

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                probability_making_decision=Pt.cpu().detach().numpy()
            )
            return loss, stats
        else:
            return loss

def calculate_probability_making_decision(deltas):
    """
    Equation 3: probability of making a decision

    :param deltas: probability of stopping at each time t
    :return: comulative probability of having stopped
    """
    batchsize, sequencelength = deltas.shape

    pts = list()

    initial_budget = torch.ones(batchsize, device=deltas.device)
    #if torch.cuda.is_available():
    #    initial_budget = initial_budget.cuda()

    budget = [initial_budget]
    for t in range(1, sequencelength):
        pt = deltas[:, t] * budget[-1]
        budget.append(budget[-1] - pt)
        pts.append(pt)

    # last time
    pt = budget[-1]
    budget.append(budget[-1] - pt)
    pts.append(pt)

    return torch.stack(pts, dim=-1)

def probability_correct_class(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).type(torch.ByteTensor).to(logprobabilities.device)

    targets_one_hot = eye[targets]

    # implement the y*\hat{y} part of the loss function
    y_haty = torch.masked_select(logprobabilities, targets_one_hot.bool())
    return y_haty.view(batchsize, seqquencelength).exp()
