# rename the files in the directory to the format of the training data
# file name format: {dataset}_{trace_id}.txt
import argparse
import os
import shutil
def main(args):
    if not os.path.isdir(args.out_dir):
	    os.mkdir(args.out_dir)
    # Iterate over each dataset directory
    for idx,file in enumerate(os.listdir(args.in_dir)):
        dataset = args.dataset
        # copy the trace file and rename it to out_dir
        shutil.copyfile(os.path.join(args.in_dir, file), os.path.join(args.out_dir, f"{dataset}_trace_{idx}.txt"))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir')
    parser.add_argument('-o', '--out-dir')
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()
    main(args)