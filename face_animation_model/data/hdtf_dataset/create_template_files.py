import os
import pickle
import glob
import tqdm



def create_template():
    tracker_out_path = "/home/bthambiraja/work/projects/dataset/HDTF"
    files_to_process = sorted(glob.glob(os.path.join(tracker_out_path, "emoca_dict", "*.pkl")))
    template_dict = {}
    for seq_to_process in tqdm.tqdm(files_to_process, desc="iterating through files"):
        out_seq_name = seq_to_process.split("/")[-1].split(".pkl")[0]
        ## load the pkl file
        with open(seq_to_process, "rb") as input_file:
            data_dict = pickle.load(input_file)
        template = data_dict["template"].view(-1, 3) # 5023 x 3
        template_dict[out_seq_name] = template

    #
    out_file = os.path.join(tracker_out_path, "templates.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(template_dict, f)


if __name__ == "__main__":
    create_template()