import os, glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import sofa
from natsort import natsorted
import librosa
from matplotlib import pyplot as plt


DATASET_PATH = "/data2/neil/HRTF/datasets/"


def transR2L(positions):
    """
    Transfer from right position to left position
    input: sequence of position: [azim, elev]
    output: indices that corresponds to the original input

    :param positions: the azim, elev of a position
    :return: the indices after transfer
    """
    new_indices = []
    for ind in range(len(positions)):
        position = positions[ind]
        new_azimuth = (360 - position[0]) % 360
        try:
            new_ind = np.where(np.isclose(positions[:, 0], new_azimuth) & np.isclose(positions[:, 1], position[1]))[0][0]
            new_indices.append(new_ind)
        except:
            continue  # print(position)

    return np.array(new_indices)


class bidict(dict):
    ## From https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


def location2degree(loc):
    azimuth, elevation = loc
    return (float("{:.2f}".format(azimuth)), float("{:.2f}".format(elevation)))  ## Note: this approximation might be reckless


class SOFADataset(Dataset):
    def __init__(self):
        super(SOFADataset, self).__init__()
        self.name = None
        self.sofa_dir = None  # a directory that include all the sofa files
        # self._expand_basic_info()

    def _expand_basic_info(self):
        self.all_sofa_files = natsorted(self._get_all_sofa_files_from_dir())
        self._get_locations_from_one_sofa(self.all_sofa_files[0])  # a list of all the locations in cartesian in degrees
        self.subject_IDs = [self._get_ID_from_sofa_path(path) for path in self.all_sofa_files]  # the id of the subjects
        self.num_of_locations = len(self.locations)  # should be equal to the length of the list of locations
        self.num_of_subjects = len(self.subject_IDs)   # the length of the list of raw subject ids
        self.num_of_ears = 2 * len(self.subject_IDs)  # should be equal to twice the num of subjects
        self.location_dict = bidict(dict(zip(self.locations, range(len(self.locations)))))
        self.subject_dict = bidict(dict(zip(self.subject_IDs, range(len(self.subject_IDs)))))

    def _get_all_sofa_files_from_dir(self):
        raise NotImplementedError()
        # return glob.glob(os.path.join(self.sofa_dir, "*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        raise NotImplementedError()

    def _get_sofa_path_from_ID(self, subject_ID):
        raise NotImplementedError()

    def _get_locations_from_one_sofa(self, path):
        self.locations = sofa.Database.open(path).Source.Position.get_values(system="spherical")
        self.locations = self.locations[:, :2]
        self.locations[:, 0] = (self.locations[:, 0] + 360) % 360
        self.locations_tensor = np.array(self.locations)
        self.locations = list(map(location2degree, self.locations.tolist()))
        self.r2l_indices = transR2L(self.locations_tensor)
        return self.locations_tensor

    def _get_location_from_locidx(self, locidx):
        return self.location_dict.inverse[locidx][0]

    def _get_locidx_from_location(self, location):
        try:
            return self.location_dict[location]
        except KeyError:
            azimuth, elevation = location
            for num, (az, el) in enumerate(self.locations):
                if math.isclose(azimuth, az, abs_tol=0.5) and math.isclose(elevation, el, abs_tol=0.8):
                    return num

    def _get_subject_idx(self, subject_ID):
        return self.subject_dict[subject_ID]

    def _get_subject_ID(self, subject_idx):
        return self.subject_dict.inverse[subject_idx][0]

    def _get_ear_ID(self, subject_ID, which_ear):
        """
        :param subject_ID: the ID of the subject, such as "003"
        :param which_ear: 0 or 1, left year is 0 and right year is 1
        :return:
        """
        subject_idx = self._get_subject_idx(subject_ID)
        return self.num_of_subjects * which_ear + subject_idx

    def __len__(self):
        return self.num_of_ears  # left and right

    def _get_frontal_locidx(self):
        return self._get_locidx_from_location((0, 0))

    def _get_HRIR(self, subject, loc, which_ear):
        if type(subject) == str:
            subject_ID = subject
            subject_idx = self._get_subject_idx(subject_ID)
        else:
            subject_idx = subject
        path = self.all_sofa_files[subject_idx]
        HRTF = sofa.Database.open(path)
        if type(loc) == tuple:
            location = loc
            measurement = self._get_locidx_from_location(location)
        else:
            measurement = loc
        if type(which_ear) == str:
            receiver = 0 if which_ear == "left" else 1
        else:
            receiver = which_ear
        emitter = 0
        # try:
        orig_ir = HRTF.Data.IR.get_values(indices={"M": measurement, "R": receiver, "E": emitter})
        # except:
        #     print(path)
        #     print(subject_idx, measurement, self._get_location_from_locidx(measurement), receiver)
        orig_sr = HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
        if orig_sr == 44100:
            return orig_ir
        else:   # resample 44100
            return librosa.resample(orig_ir, int(orig_sr), 44100)

    def _sanity_check(self):
        for key in list(self.__dir__()):
            if not key.startswith("_"):
                try:
                    assert not self.__dict__[key] is None
                except:
                    print("%s is not set appropriately." % key)

    def _print_basic_info(self):
        print("\nDataset:", self.name.upper())
        self._sanity_check()
        print("Number of subjects", self.num_of_subjects)
        print("Number of locations", self.num_of_locations)
        print("How many locations are symmetrical?", len(self.r2l_indices))
        print("Frontal direction index", self._get_frontal_locidx())
        print(min(self.locations_tensor[:, 1]), max(self.locations_tensor[:, 1]))
        # print(self.locations)

    def _plot_frontal_HRIR(self, subject_idx, which_ear, ax):
        ir = self._get_HRIR(subject_idx, (0, 0), which_ear)
        ir = ir[:500]
        ax.plot(ir)
        ax.set(xticks=list(np.arange(0, 600, 100)),
                xticklabels=['{:,.5f}'.format(x) for x in list(np.arange(0, 600, 100) / 44100)],
                title="Frontal HRIR from %s, Subject:%s" % (self.name.upper(), self._get_subject_ID(subject_idx)),
                ylabel='Magnitude',
                xlabel='Time (s)')

    def _plot_frontal_HRTF(self, subject_idx, which_ear, ax):
        ir = self._get_HRIR(subject_idx, (0, 0), which_ear)
        # ir = ir[:500]
        tf = np.fft.fft(ir, n=256)
        ax.plot(np.log(np.abs(tf[:128])))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
                xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
                title="Frontal HRTF from %s, Subject:%s" % (self.name.upper(), self._get_subject_ID(subject_idx)),
                ylabel='Log Magnitude',
                xlabel='Frequency (Hz)')

    def _plot_midsagittal_plane_linear(self, subject_idx, which_ear, ax):
        ## get the hrtf of azimuth=0
        size_ = 20
        selected_locs = []
        midsagittal_hrtfs = []
        for loc in self.locations_tensor:
            if loc[0] == 0 or loc[0] == 180:
                selected_locs.append(loc)
                ir = self._get_HRIR(subject_idx, tuple(loc), which_ear)
                tf = np.abs(np.fft.fft(ir, n=256))
                midsagittal_hrtfs.append(tf[:128])
        selected_locs = np.array(selected_locs)
        ## sort the elevation and get the argsort
        sorted_args = np.argsort(selected_locs[:, 1] - selected_locs[:, 0] / 90 * selected_locs[:, 1] + selected_locs[:, 0])
        ## rearange the hrtf
        midsagittal_hrtfs = np.array(midsagittal_hrtfs)[sorted_args]
        midsagittal_hrtfs = midsagittal_hrtfs / np.max(midsagittal_hrtfs)
        midsagittal_hrtfs = np.log(midsagittal_hrtfs)
        img = ax.pcolormesh(midsagittal_hrtfs, cmap=plt.cm.viridis)
        # title = "Midsagittal Plane from %s, Subject:%s" % (self.name.upper(), self._get_subject_ID(subject_idx))
        img.set_clim(-6, 0)
        cbar = ax.figure.colorbar(img, ax=ax, ticks=[-6, -3, 0])
        cbar.ax.tick_params(labelsize=size_)
        ax.set(xticks=list(np.arange(3, 92 + 16, 16)),
            yticks=np.arange(0, selected_locs.shape[0], 1),
            xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(3, 92 + 16, 16) * 44.1 / 256)],
            yticklabels=['{:,.2f}'.format(x) for x in selected_locs[sorted_args, 1].tolist()],
            # title=title,
            ylabel='Polar angle (degree)',
            xlabel='Frequency (Hz)')
        ax.xaxis.get_label().set_fontsize(size_)
        ax.yaxis.get_label().set_fontsize(size_)
        return midsagittal_hrtfs.shape[0]

    def __getitem__(self, idx):
        """
        :param idx: Here the idx is the idx of the ear, from it we can know the subject_idx and which_ear
        :return: locations and HRIRs
        """
        subject_idx = idx // 2
        which_ear = idx % 2
        sofa_path = self.all_sofa_files[subject_idx]
        HRTF = sofa.Database.open(sofa_path)
        locations = HRTF.Source.Position.get_values(system="spherical")
        locations = locations[:, :2]
        if which_ear == 1:  # right ear map to the left
            locations[:, 0] = 360 - locations[:, 0]
        locations[:, 0] = (locations[:, 0] + 360) % 360
        irs = []
        for measurement in range(len(locations)):
            receiver = which_ear
            emitter = 0
            # try:
            orig_ir = HRTF.Data.IR.get_values(indices={"M": measurement, "R": receiver, "E": emitter})
            # except:
            #     print(sofa_path)
            #     print(subject_idx, measurement, self._get_location_from_locidx(measurement), receiver)
            orig_sr = HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
            if orig_sr == 44100:
                irs.append(np.array(orig_ir))
            else:  # resample 44100
                irs.append(librosa.resample(orig_ir, int(orig_sr), 44100))
        return locations, np.array(irs)


class ARI(SOFADataset):
    def __init__(self):
        super(ARI, self).__init__()
        self.name = "ari"
        self.sofa_dir = os.path.join(DATASET_PATH, "ari")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "hrtf_nh*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1][2:].split(".")[0]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "hrtf_nh%s.sofa" % subject_ID)


class HUTUBS(SOFADataset):
    def __init__(self):
        super(HUTUBS, self).__init__()
        self.name = "hutubs"
        self.sofa_dir = os.path.join(DATASET_PATH, "HUTUBS/HRIRs")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "*HRIRs_measured.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][2:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "pp%d_HRIRs_measured.sofa" % int(subject_ID))


class ITA(SOFADataset):
    def __init__(self):
        super(ITA, self).__init__()
        self.name = "ita"
        self.sofa_dir = os.path.join(DATASET_PATH, "ITA/SOFA")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "MRT*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split(".")[0][3:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "MRT%02d.sofa" % int(subject_ID))


class CIPIC(SOFADataset):
    def __init__(self):
        super(CIPIC, self).__init__()
        self.name = "cipic"
        self.sofa_dir = os.path.join(DATASET_PATH, "CIPIC/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "subject_*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1][:3]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "subject_%03d.sofa" % int(subject_ID))


class SADIE(SOFADataset):
    def __init__(self):
        super(SADIE, self).__init__()
        self.name = "sadie"
        self.sofa_dir = os.path.join(DATASET_PATH, "SADIE/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "H*_44K_16bit_256tap_FIR_SOFA.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][1:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "H%s_44K_16bit_256tap_FIR_SOFA.sofa" % subject_ID)


class Prin3D3A(SOFADataset):
    def __init__(self):
        super(Prin3D3A, self).__init__()
        self.name = "3d3a"
        self.sofa_dir = os.path.join(DATASET_PATH, "3d3a")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "Subject*_HRIRs.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][7:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "Subject%s_HRIRs.sofa" % subject_ID)


class RIEC(SOFADataset):
    def __init__(self):
        super(RIEC, self).__init__()
        self.name = "riec"
        self.sofa_dir = os.path.join(DATASET_PATH, "riec")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "RIEC_hrir_subject_*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[-1][:3]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "RIEC_hrir_subject_%s.sofa" % subject_ID)


class Listen(SOFADataset):
    def __init__(self):
        super(Listen, self).__init__()
        self.name = "listen"
        self.sofa_dir = os.path.join(DATASET_PATH, "listen")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "IRC_*_C_44100.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_44100.sofa" % subject_ID)


class Crossmod(SOFADataset):
    def __init__(self):
        super(Crossmod, self).__init__()
        self.name = "crossmod"
        self.sofa_dir = os.path.join(DATASET_PATH, "crossmod")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "IRC_*_C_44100.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_44100.sofa" % subject_ID)


class BiLi(SOFADataset):
    def __init__(self):
        super(BiLi, self).__init__()
        self.name = "bili"
        self.sofa_dir = os.path.join(DATASET_PATH, "bili")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "IRC_*_C_HRIR_96000.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_HRIR_96000.sofa" % subject_ID)


class MergedSOFADataset(Dataset):
    def __init__(self):
        self.all_dataset_funcs = [ARI, HUTUBS, ITA, CIPIC, Prin3D3A, RIEC, BiLi, Listen, Crossmod]
        self.all_datasets = []
        self.length_array = []
        for dataset_func in self.all_dataset_funcs:
            self.all_datasets.append(dataset_func())
        for dataset in self.all_datasets:
            self.length_array.append(len(dataset))
            irs_array = []
            for i in range(0, len(dataset), 2):
                location, irs = dataset[i]
                irs_array.append(irs)
            irs_array = np.array(irs_array)
            dataset.mean_left = np.mean(irs_array, axis=0)
            dataset.std_left = np.std(irs_array, axis=0)
            irs_array = []
            for i in range(1, len(dataset), 2):
                location, irs = dataset[i]
                irs_array.append(irs)
            irs_array = np.array(irs_array)
            dataset.mean_right = np.mean(irs_array, axis=0)
            dataset.std_right = np.std(irs_array, axis=0)

    def __len__(self):
        return np.sum(self.length_array)

    def __getitem__(self, idx):
        # Given idx, find which dataset
        length_sum = np.cumsum(self.length_array)
        for j in range(len(length_sum)):
            if idx < length_sum[j]:
                dataset_idx = j
                item_idx = idx - length_sum[j-1]
                break
        ## normalize their magnitude
        dataset = self.all_datasets[dataset_idx]
        locs, irs = dataset[item_idx]
        if item_idx % 2 == 0:
            irs = (irs - dataset.mean_left) / dataset.std_left
        else:
            irs = (irs - dataset.mean_right) / dataset.std_right
        return locs, irs




if __name__ == "__main__":
    # hutubs = HUTUBS()
    # hutubs.check_valid()
    # print(len(hutubs))
    # for i in range(len(hutubs)):
    #     hrtf, subject, freq, left_or_right = hutubs[i]
    master_dataset = SOFADataset()
    master_dataset._sanity_check()




