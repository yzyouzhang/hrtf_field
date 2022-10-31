from HRTFdatasets import MergedHRTFDataset, PartialHRTFDataset
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import argparse
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    """
    set initial seed for reproduction
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def gon_model(dimensions):
    first_layer = SirenLayer(dimensions[0], dimensions[1], is_first=True)
    other_layers = []
    # other_layers.append(nn.LayerNorm(dimensions[1]))
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        other_layers.append(SirenLayer(dim0, dim1))
        # other_layers.append(nn.LayerNorm(dim1))
    final_layer = SirenLayer(dimensions[-2], dimensions[-1], is_last=True)
    return nn.Sequential(first_layer, *other_layers, final_layer)


def keras_decay(step, decay=0.01):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)

def twoD2threeD(locs, hrtfs):
    if not type(locs) is np.ndarray:
        locs = locs.numpy()
        hrtfs = hrtfs.numpy()
    index = np.where(np.logical_and(locs[:, 0] < 360, locs[:, 0] >= 0))[0]
    # convert 2d location to cartesian
    # loc: azi, ele
    if hrtfs.shape[1] == 1:
        hrtfs = np.squeeze(hrtfs)[index]
    else:
        rand_f = 15  # np.random.randint(hrtfs.shape[1])
        hrtfs = hrtfs[index, rand_f]
    azi = locs[index, 0] / 180 * np.pi
    ele = locs[index, 1] / 180 * np.pi
    x = hrtfs * np.cos(ele) * np.cos(azi)
    y = hrtfs * np.cos(ele) * np.sin(azi)
    z = hrtfs * np.sin(ele)
    xyz = np.array([x,y,z]).T
    return xyz

def plot_hrtf_3D(locs, hrtfs, note="", folder="./"):
    xyz = twoD2threeD(locs, hrtfs)
    df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    df['r'] = np.sqrt(np.sum(xyz ** 2, axis=1))
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        opacity=0.8, color='r',
                        height=900, width=950)
    fig.write_html(os.path.join(folder, note+"_3D_HRTF.html"))
    return fig


def plot_hrtf_2D(locs, hrtfs, mask, ax, c='b', label=""):
    index = np.where(np.logical_and(locs[:, 0] < 360, locs[:, 0] >= 0))[0]
    # convert 2d location to cartesian
    # loc: azi, ele
    if hrtfs.shape[1] == 1:
        hrtfs = np.squeeze(hrtfs)[index]
    else:
        rand_f = 15  # np.random.randint(hrtfs.shape[1])
        hrtfs = hrtfs[index, rand_f]
    num_loc = index.shape[0]
    ax.plot(np.arange(num_loc), hrtfs[:num_loc], c, label=label)
    return ax


def gon_sample(model, recent_zs, coords, batch_size, device):
    zs = torch.cat(recent_zs, dim=0).squeeze(1).cpu().numpy()
    mean = np.mean(zs, axis=0)
    cov = np.cov(zs.T)
    sample = np.random.multivariate_normal(mean, cov, size=batch_size)
    sample = torch.tensor(sample).unsqueeze(1).repeat(1,coords.shape[1],1).to(device).float()
    model_input = torch.cat((coords, sample), dim=-1)
    return model(model_input)


def metrics(gt, pred, masks, scale="linear"):
    if scale == "linear":
        lsd_elements = torch.square(20 * torch.log10(torch.abs(gt) / torch.abs(pred)))
    elif scale == "log":
        lsd_elements = torch.square(gt - pred)
    else:
        raise ValueError("Either log or linear scale")
    square_sum = (lsd_elements * masks).sum()
    mask_sum = masks.sum()
    lsd = square_sum / mask_sum
    return torch.sqrt(lsd), square_sum.item(), mask_sum.item()


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=66)

    # Data folder prepare
    parser.add_argument("-o", "--out_fold", type=str, help="output folder",
                        required=True, default='/data/neil/hrtf_field/models0913')

    # Dataset parameters
    parser.add_argument("-d", "--dataset_path", type=str, default="/data2/neil/HRTF/datasets/")
    parser.add_argument('-n', '--training_dataset_names', nargs='+',
                        default=["ari", "hutubs", "cipic", "3d3a", "ita",
                                 "bili", "listen", "crossmod", "sadie"])
    parser.add_argument('-t', '--testing_dataset_names', nargs='+',
                        default=["riec"])
    parser.add_argument("-f", "--frequency_idx", type=int, default=1000, help="index of frequency of HRTF, 1000 if use all")
    parser.add_argument("-s", "--scale", choices=["log", "linear"],
                        default="log", help="magnitude in the log or linear scale")
    parser.add_argument("--norm_way", type=int, default=2, help="way of normalization across datasets")

    # Model parameters
    parser.add_argument("-w", "--first_w0", type=float, default=30, help="w0 for the first SIREN layer")
    parser.add_argument("-z", "--num_latent", type=int,  default=32, help="latent code dimension")
    parser.add_argument('--hidden_features', type=int, default=2048, help="hidden layer dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of hidden layers")

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=18, help="Mini batch size for training")
    parser.add_argument('--num_workers', type=int, default=9, help="number of workers")
    parser.add_argument("--lr", type=float, default=3 * 1e-4, help="adam learning rate")
    parser.add_argument("--decay", type=float, default=0.01, help="learning rate decay as keras style")

    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--interval', type=int, default=3, help="epoch interval for plot check")

    args = parser.parse_args()

    if "all" in args.training_dataset_names:
        args.training_dataset_names = ["ari", "hutubs", "cipic", "3d3a", "riec",
                                       "bili", "listen", "crossmod", "sadie", "ita"]

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    set_seed(args.seed)

    if args.frequency_idx == 1000:  # use 1000 to indicate you want to use all frequency
        args.frequency_idx = "all"

    # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        print("The output folder has already existed, please change another folder")

    # Path for input data
    assert os.path.exists(args.dataset_path)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    args.cuda = torch.cuda.is_available() and int(args.gpu) >= 0
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def train_one_epoch(trainDataLoader, F, recent_zs, optim, lr_scheduler, args):
    device = args.device
    F.train()
    square_sums, mask_sums = 0, 0
    for i, (locs, hrtfs, masks, names) in enumerate(trainDataLoader):
        # sample a batch of data and to device
        locs = locs.to(device)
        hrtfs = hrtfs.to(device)
        masks = masks.to(device)

        c = locs.float()
        x = hrtfs

        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = torch.zeros(locs.shape[0], 1, args.num_latent).to(device).requires_grad_()
        z_rep = z.repeat(1, c.size(1), 1)
        g = F(torch.cat((c, z_rep), dim=-1))
        L_inner = (((g - x) ** 2) * masks).sum() / masks.sum()
        z = -torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]

        # # one more step of gon for non-linearity
        # z_rep = z.repeat(1, c.size(1), 1)
        # g = F(torch.cat((c, z_rep), dim=-1))
        # L_inner = (((g - x) ** 2) * masks).sum() / masks.sum()
        # z = z - torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]

        # now with z as our new latent points, optimise the data fitting loss
        z_rep = z.repeat(1, c.size(1), 1)
        g = F(torch.cat((c, z_rep), dim=-1))
        L_outer = (((g - x) ** 2) * masks).sum() / masks.sum()
        optim.zero_grad()
        L_outer.backward()
        optim.step()
        lsd, square_sum, mask_sum = metrics(x, g, masks, args.scale)
        wandb.log({"l_out": L_outer.item(), "lsd": lsd.item()})

        # compute sampling statistics
        recent_zs.append(z.detach())
        recent_zs = recent_zs[-30:]

        square_sums += square_sum
        mask_sums += mask_sum
    lr_scheduler.step()

    training_lsd = np.sqrt(square_sums / mask_sums)

    return F, training_lsd, c, x, g, masks, names, recent_zs, optim, lr_scheduler


def plot_epoch(epoch, F, c, x, g, masks, names, recent_zs, args, subfolder='checkpoint'):
    device = args.device
    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, subfolder)):
        os.makedirs(os.path.join(args.out_fold, subfolder))
    if args.scale == "log":
        g = torch.pow(10, g / 20)
        x = torch.pow(10, x / 20)

    # Reconstruct HRTF
    fig = plot_hrtf_3D(c[0].detach().cpu().numpy(), g[0].detach().cpu().numpy(),
                       "%03dep_Reconstructed_%s" % (epoch, names[0]), os.path.join(args.out_fold, subfolder))
    fig = plot_hrtf_3D(c[0].detach().cpu().numpy(), x[0].detach().cpu().numpy(),
                       "%03dep_GroundTruth_%s" % (epoch, names[0]), os.path.join(args.out_fold, subfolder))
    fig = plot_hrtf_3D(c[0].detach().cpu().numpy(), (g[0] - x[0]).detach().cpu().numpy(),
                       "%03dep_Error_%s" % (epoch, names[0]), os.path.join(args.out_fold, subfolder))

    fig, ax = plt.subplots(1, figsize=(16, 6))
    ax = plot_hrtf_2D(c[0].detach().cpu().numpy(), x[0].detach().cpu().numpy(), masks[0].detach().cpu().numpy(), ax,
                      c="r", label="GT")
    ax = plot_hrtf_2D(c[0].detach().cpu().numpy(), g[0].detach().cpu().numpy(), masks[0].detach().cpu().numpy(), ax,
                      c="b", label="Recon")
    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_fold, subfolder, '%03dep_reconstruction_%s.png' % (epoch, names[0])), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def test_one_epoch(testDataLoader, F, args):
    device = args.device
    F.eval()
    square_sums, mask_sums = 0, 0
    for i, (locs, hrtfs, masks, names) in enumerate(testDataLoader):
        locs = locs.to(device)
        hrtfs = hrtfs.to(device)
        masks = masks.to(device)

        c = locs.float()
        x = hrtfs

        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = torch.zeros(locs.shape[0], 1, args.num_latent).to(device).requires_grad_()
        z_rep = z.repeat(1, c.size(1), 1)
        g = F(torch.cat((c, z_rep), dim=-1))
        L_inner = (((g - x) ** 2) * masks).sum() / masks.sum()
        z = -torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]

        with torch.no_grad():
            z_rep = z.repeat(1, c.size(1), 1)
            g = F(torch.cat((c, z_rep), dim=-1))
            lsd, square_sum, mask_sum = metrics(x, g, masks, args.scale)
            square_sums += square_sum
            mask_sums += mask_sum
    testing_lsd = np.sqrt(square_sums / mask_sums)
    return testing_lsd, F, c, x, g, names


def run(args):
    trainDataset = MergedHRTFDataset(args.training_dataset_names,
                                     args.frequency_idx, args.scale, args.norm_way)
    trainDataLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=trainDataset.collate_fn, num_workers=args.num_workers)
    # partialTrainDataset = PartialHRTFDataset(args.testing_dataset_names, args.frequency_idx, args.scale, args.norm_way)
    # trainDataLoader = DataLoader(trainDataset + partialTrainDataset, batch_size=args.batch_size, shuffle=True,
    #                              collate_fn=trainDataset.collate_fn, num_workers=args.num_workers)
    testDataset = MergedHRTFDataset(args.testing_dataset_names,
                                    args.frequency_idx, args.scale, args.norm_way)
    testDataLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=testDataset.collate_fn, num_workers=args.num_workers)
    # Original HRTF
    for j in range(10):
        rand_idx = np.random.randint(len(trainDataset))
        loc, hrtf, name = trainDataset[rand_idx]
        if args.scale == "log":
            hrtf = np.power(10, hrtf / 20)
        fig = plot_hrtf_3D(loc, hrtf, "Original%03d_%s" % (rand_idx, name), args.out_fold)

    coords = loc.shape[1]
    n_channels = hrtf.shape[1]
    device = args.device
    # define GON architecture
    gon_shape = [coords + args.num_latent] + [args.hidden_features] * args.num_layers + [n_channels]
    F = gon_model(gon_shape).to(device)
    optim = torch.optim.Adam(lr=args.lr, params=F.parameters(), weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: keras_decay(step, args.decay)
    )
    print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')


    prev_lsd = 1e6
    early_stop = 0
    recent_zs = []
    for epoch in tqdm(range(args.num_epochs)):
        # print("Epoch: %03d" % epoch)
        if early_stop > 30:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (best_epoch))
            # break
        F, training_lsd, c, x, g, masks, names, recent_zs, \
        optim, lr_scheduler = train_one_epoch(trainDataLoader, F, recent_zs,
                                              optim, lr_scheduler, args)
        wandb.log({"training_lsd": training_lsd})

        if epoch % args.interval == 0:
            plot_epoch(epoch, F, c, x, g, masks, names, recent_zs, args)
            torch.save(F.state_dict(),
                       os.path.join(args.out_fold, 'checkpoint',
                                    'gon_epoch%03d_lsd%.3f.pt' % (epoch, training_lsd)))

        testing_lsd, F, c, x, g, names = test_one_epoch(testDataLoader, F, args)
        wandb.log({"testing_lsd": testing_lsd})

        if testing_lsd < prev_lsd:
            prev_lsd = testing_lsd
            best_epoch = epoch
            early_stop = 0
            torch.save(F.state_dict(),
                       os.path.join(args.out_fold, 'gon_lsd%.3f.pt' % (testing_lsd)))
        else:
            early_stop += 1

    print("Best testing LSD:", prev_lsd)

    return F, testing_lsd



if __name__ == "__main__":
    import wandb

    os.environ["WANDB_API_KEY"] = "ad172f7793efc7ce6fc853de46ef015d6f1769cf"
    # wandb.login()
    args = initParams()
    wandb.init(project="hrtf_siren",
               entity="yzyouzhang",
               name=os.path.basename(args.out_fold),
               config=args)
    F, testing_lsd = run(args)
    wandb.finish()


