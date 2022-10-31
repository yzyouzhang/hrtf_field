import os, glob
import pickle as pkl
import SOFAdatasets
from tqdm import tqdm


out_dir = "/data2/neil/HRTF/prepocessed_hrirs"


def save_hrir_as_pkl():
    dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                    "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                    "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
    for name in list(dataset_dict.keys()):
        # name = "riec"
        print(name)
        dataset_obj = getattr(SOFAdatasets, dataset_dict[name])()
        for idx in tqdm(range(len(dataset_obj))):
            location, hrir = dataset_obj[idx]
            filename = "%s_%03d.pkl" % (name, idx)
            with open(os.path.join(out_dir, filename), 'wb') as handle:
                pkl.dump((location, hrir), handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    save_hrir_as_pkl()

