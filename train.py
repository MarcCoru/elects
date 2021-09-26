from data import BavarianCrops, BreizhCrops
from torch.utils.data import DataLoader
from earlyrnn import EarlyRNN
import torch
from tqdm import tqdm
from loss import EarlyRewardLoss
import numpy as np
from utils import VisdomLogger
import sklearn.metrics
import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run ELECTS Early Classification training on the BavarianCrops dataset.')
    parser.add_argument('--dataset', type=str, default="bavariancrops", choices=["bavariancrops","breizhcrops"], help="dataset")
    parser.add_argument('--alpha', type=float, default=0.5, help="trade-off parameter of earliness and accuracy (eq 6): "
                                                                 "1=full weight on accuracy; 0=full weight on earliness")
    parser.add_argument('--epsilon', type=float, default=10, help="additive smoothing parameter that helps the "
                                                                  "model recover from too early classificaitons (eq 7)")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda","cpu"], help="'cuda' (GPU) or 'cpu' device to run the code. "
                                                     "defaults to 'cuda' if GPU is available, otherwise 'cpu'")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--sequencelength', type=int, default=70, help="sequencelength of the time series. If samples are shorter, "
                                                                "they are zero-padded until this length; "
                                                                "if samples are longer, they will be undersampled")
    parser.add_argument('--batchsize', type=int, default=256, help="number of samples per batch")
    parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ["HOME"],"elects_data"), help="directory to download the "
                                                                                 "BavarianCrops dataset (400MB)."
                                                                                 "Defaults to home directory.")
    parser.add_argument('--snapshot', type=str, default="snapshots/model.pth",
                        help="pytorch state dict snapshot file")

    return parser.parse_args()

def main(args):

    if args.dataset == "bavariancrops":
        dataset_class = BavarianCrops
        dataroot = os.path.join(args.dataroot,"bavariancrops")
        nclasses = 7
    elif args.dataset == "breizhcrops":
        dataset_class = BreizhCrops
        dataroot = os.path.join(args.dataroot,"breizhcrops")
        nclasses = 9

    traindataloader = DataLoader(
        dataset_class(root=dataroot,partition="train", sequencelength=args.sequencelength),
        batch_size=args.batchsize)
    testdataloader = DataLoader(
        dataset_class(root=dataroot,partition="valid", sequencelength=args.sequencelength),
        batch_size=args.batchsize)

    model = EarlyRNN(nclasses=nclasses).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)

    visdom_logger = VisdomLogger()

    with tqdm(range(1, args.epochs + 1)) as pbar:
        train_stats = []
        for epoch in pbar:
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=args.device)
            testloss, stats = test_epoch(model, testdataloader, criterion, args.device)

            # statistic logging and visualization...
            precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0], average="macro",
                zero_division=0)
            accuracy = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0])
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0])

            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))

            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_pred=stats["predictions_at_t_stop"][:, 0],
                                                                         y_true=stats["targets"][:, 0])

            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward
                )
            )

            visdom_logger(stats)
            visdom_logger.plot_boxplot(stats["targets"][:, 0], stats["t_stop"][:, 0], tmin=0, tmax=args.sequencelength)
            df = pd.DataFrame(train_stats).set_index("epoch")
            visdom_logger.plot_epochs(df[["precision", "recall", "fscore", "kappa"]], name="accuracy metrics")
            visdom_logger.plot_epochs(df[["trainloss", "testloss"]], name="losses")
            visdom_logger.plot_epochs(df[["accuracy", "earliness"]], name="accuracy, earliness")
            visdom_logger.plot_epochs(df[["classification_loss", "earliness_reward"]], name="loss components")

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                                 f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                                 f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}")

    print(f"saving model to {args.snapshot}")
    os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
    torch.save(model.state_dict(), args.snapshot)
    df.to_csv(args.snapshot + ".csv")


def train_epoch(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)
        log_class_probabilities, probability_stopping = model(X)

        loss = criterion(log_class_probabilities, probability_stopping, y_true)

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())

    return np.stack(losses).mean()

def test_epoch(model, dataloader, criterion, device):
    model.eval()

    stats = []
    losses = []
    for batch in dataloader:
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
        loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)

        stat["loss"] = loss.cpu().detach().numpy()
        stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
        stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
        stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["targets"] = y_true.cpu().detach().numpy()

        stats.append(stat)

        losses.append(loss.cpu().detach().numpy())

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}

    return np.stack(losses).mean(), stats

if __name__ == '__main__':
    args = parse_args()
    main(args)
