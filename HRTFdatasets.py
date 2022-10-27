import SOFAdatasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import pickle as pkl
import os


class HRTFDataset(Dataset):
    def __init__(self, dataset="crossmod", freq=15, scale="linear", norm_way=0):
        ## assert dataset is one of HRTFDataset
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
        self.name = dataset
        self.dataset_obj = getattr(SOFAdatasets, dataset_dict[self.name])()
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        # self.max_mag = self._find_global_max_magnitude()

    def __len__(self):
        return self.dataset_obj.__len__()

    def _get_hrtf(self, idx, freq, scale="linear", norm_way=0):
        # location, hrir = self.dataset_obj[idx]
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            location, hrir = pkl.load(handle)
        tf = np.abs(np.fft.fft(hrir, n=256))
        tf = tf[:, 1:93]  # first 128 freq bins, but up to 16k
        # tf = tf[:, 3:93]   # 500 Hz to 16kHz contribute to localization and are equalized
        ## how to normalize
        ## first way is to devide by max value
        if norm_way == 0:
            tf = tf / np.max(tf)
        ## second way is to devide by top 5% top value
        elif norm_way == 1:
            mag_flatten = tf.flatten()
            max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
            tf = tf / max_mag
        ## third way is to compute total energy of the equator
        elif norm_way == 2:
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            tf_equator = tf[equator_index]
            equator_azi = location[equator_index, 0][0]
            new_equator_index = np.argsort(equator_azi)
            new_equator_azi = equator_azi[new_equator_index]
            new_equator_tf = tf_equator[new_equator_index]

            total_energy = 0
            for x in range(len(new_equator_index)):
                if x == 0:
                    d_azi = 360 - new_equator_azi[-1]
                    # d_azi = new_equator_azi[1] - new_equator_azi[0]
                else:
                    d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                total_energy += np.square(new_equator_tf[x]).mean() * d_azi
            tf = tf / np.sqrt(total_energy / 360)
            # print(np.sqrt(total_energy / 360))
        ## fourth way is to normalize on common locations
        ## [(0.0, 0.0), (180.0, 0.0), (210.0, 0.0), (330.0, 0.0), (30.0, 0.0), (150.0, 0.0)]
        elif norm_way == 3:
            common_index = np.where(np.logical_and(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0),
                                                   np.array(
                                                       [round(x) in [0, 180, 210, 330, 30, 150] for x in location[:, 0]])))
            tf_common = tf[common_index]
            mean_energy = np.sqrt(np.square(tf_common).mean())
            # print(mean_energy)
            tf = tf / mean_energy

        if scale == "linear":
            tf = tf
        elif scale == "log":
            tf = 20 * np.log10(tf)
        if freq == "all":
            return location, tf
        return location, tf[:, freq][:, np.newaxis]

    def _find_global_max_magnitude(self):
        max_mag = 0
        for i in range(self.__len__()):
            _, tf_mag = self._get_hrtf(i, "all", "linear")
            cur_max_mag = np.max(tf_mag)
            if cur_max_mag > max_mag:
                max_mag = cur_max_mag
        return max_mag

    def __getitem__(self, idx):
        location, hrtf = self._get_hrtf(idx, self.freq, self.scale, self.norm_way)
        # return location, hrtf / self.max_mag
        return location, hrtf

    def _plot_frontal_data(self, idx, ax):
        loc_idx = self.dataset_obj._get_frontal_locidx()
        _, hrtf = self._get_hrtf(idx, "all", "linear")
        # hrtf = hrtf / self.max_mag
        ax.plot(np.log(hrtf[loc_idx]), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title="Frontal HRTF",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')


class MergedHRTFDataset(Dataset):
    def __init__(self, all_dataset_names, freq, scale="linear", norm_way=2):
        self.all_dataset_names = all_dataset_names
        # ["ari", "hutubs", "cipic", "3d3a", "riec", "bili", "listen", "crossmod", "sadie", "ita"]
        self.all_datasets = []
        self.length_array = []
        self.all_data = []
        for dataset_name in self.all_dataset_names:
            self.all_datasets.append(HRTFDataset(dataset_name, freq, scale, norm_way))
        for dataset in self.all_datasets:
            for item_idx in range(len(dataset)):
                locs, hrtfs = dataset[item_idx]
                self.all_data.append((locs, hrtfs, dataset.name))
            self.length_array.append(len(dataset))
        # self.length_sum = np.insert(np.cumsum(self.length_array), 0, 0)

    def __len__(self):
        return np.sum(self.length_array)

    def extend_locations(self, locs, hrtfs):
        ## Extend locations from -30 to 0 and from 360 to 390
        index1 = np.where(locs[:, 0] > 330)
        new_locs1 = locs.copy()[index1]
        new_locs1[:, 0] -= 360
        index2 = np.where(locs[:, 0] < 30)
        new_locs2 = locs.copy()[index2]
        new_locs2[:, 0] += 360
        num_loc = new_locs1.shape[0] + locs.shape[0] + new_locs2.shape[0]
        # assign values for locs
        new_locs = torch.zeros(num_loc, 2)
        new_locs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(locs)
        new_locs[:new_locs1.shape[0]] = torch.from_numpy(new_locs1)
        new_locs[-new_locs2.shape[0]:] = torch.from_numpy(new_locs2)
        # assign values for hrtfs
        new_hrtfs = torch.zeros(num_loc, hrtfs.shape[1])
        new_hrtfs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(hrtfs)
        new_hrtfs[:new_locs1.shape[0]] = torch.from_numpy(hrtfs.copy()[index1])
        new_hrtfs[-new_locs2.shape[0]:] = torch.from_numpy(hrtfs.copy()[index2])
        return new_locs, new_hrtfs

    def __getitem__(self, idx):
        locs, hrtfs, names = self.all_data[idx]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names

    def collate_fn(self, samples):
        B = len(samples)
        len_sorted, _ = torch.sort(torch.Tensor([sample[0].shape[0] for sample in samples]), descending=True)
        max_num_loc = int(len_sorted[0].item())
        n_freq = samples[0][1].shape[1]
        locs, hrtfs, masks, names = [], [], [], []
        locs = -torch.ones((B, max_num_loc, 2))
        hrtfs = -torch.ones((B, max_num_loc, n_freq))
        masks = torch.zeros((B, max_num_loc, n_freq))
        for idx, sample in enumerate(samples):
            num_loc = sample[0].shape[0]
            loc, hrtf, name = sample
            locs[idx, :num_loc, :] = loc
            hrtfs[idx, :num_loc, :] = hrtf
            masks[idx, :num_loc, :] = 1
            names.append(name)
        return locs, hrtfs, masks, default_collate(names)


class PartialHRTFDataset(MergedHRTFDataset):
    def __init__(self, dataset_name="riec", freq=15, scale="linear", norm_way=2):
        super().__init__(dataset_name, freq, scale, norm_way)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        locs, hrtfs, names = self.all_data[idx]
        indices = np.array([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
        26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
        52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  73,  75,  77,
        79,  81,  83,  85,  87,  89,  91,  93,  95,  97,  99, 101, 103,
       105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129,
       131, 133, 135, 137, 139, 141, 143, 144, 146, 148, 150, 152, 154,
       156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
       182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206,
       208, 210, 212, 214, 217, 219, 221, 223, 225, 227, 229, 231, 233,
       235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259,
       261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285,
       287, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
       312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336,
       338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 361, 363,
       365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389,
       391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415,
       417, 419, 421, 423, 425, 427, 429, 431, 432, 434, 436, 438, 440,
       442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466,
       468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492,
       494, 496, 498, 500, 502, 505, 507, 509, 511, 513, 515, 517, 519,
       521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545,
       547, 549, 551, 553, 555, 557, 559, 561, 563, 565, 567, 569, 571,
       573, 575, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596,
       598, 600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622,
       624, 626, 628, 630, 632, 634, 636, 638, 640, 642, 644, 646, 649,
       651, 653, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675,
       677, 679, 681, 683, 685, 687, 689, 691, 693, 695, 697, 699, 701,
       703, 705, 707, 709, 711, 713, 715, 717, 719, 720, 722, 724, 726,
       728, 730, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752,
       754, 756, 758, 760, 762, 764, 766, 768, 770, 772, 774, 776, 778,
       780, 782, 784, 786, 788, 790, 793, 795, 797, 799, 801, 803, 805,
       807, 809, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831,
       833, 835, 837, 839, 841, 843, 845, 847, 849, 851, 853, 855, 857,
       859, 861, 863, 864])
        locs = locs[indices]
        hrtfs = hrtfs[indices]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names


class HRTFFitting(Dataset):
    def __init__(self, location, hrtf, part="full"):
        super(HRTFFitting, self).__init__()
        assert part in ["full", "half", "random_half"]
        num_locations = location.shape[0]
        assert hrtf.shape[0] == num_locations

        self.hrtf = hrtf
        self.coords = location

        if part == "full":
            self.indices = np.arange(num_locations)
        elif part == "half":
            self.indices = np.arange(0, num_locations, 2)
        elif part == "random_half":
            self.indices = np.random.choice(num_locations, num_locations // 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords[self.indices], self.hrtf[self.indices]


def fitting_dataset_wrapper(idx, dataset="crossmod", freq=1, part="full"):
    dataset = HRTFDataset(dataset, freq)
    loc, hrtf = dataset[idx]
    return HRTFFitting(loc, hrtf, part)



if __name__ == "__main__":
    res = HRTFDataset()
    loc, hrtf = res[3]
    print(loc.shape)
    print(hrtf.shape)



