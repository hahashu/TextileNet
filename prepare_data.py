import argparse
import json
import argparse
import os
import requests


parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--parent_dir', type=str, help='parent_directory')
parser.add_argument('--partition', type=str, help='fibre or fabric')
parser.add_argument('--test', action='store_true')
parser.add_argument('--processes', type=int, help='number of processes', default=None)
args = parser.parse_args()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def download_files(parent_dir, files):
    for fname, cls, url in files:
        img_data = requests.get(url).content
        with open(f'{parent_dir}/{cls}/{fname}', 'wb') as handler:
            handler.write(img_data)


print(args.parent_dir, args.partition, args.test)

test_data = read_json(f'json/{args.partition}_test.json')
classes = [i[1] for i in test_data]
classes = list(set(classes))

for cls in classes:
    # make dir and ignore if it exists
    os.makedirs(f'{args.parent_dir}/train/{cls}', exist_ok=True)
    os.makedirs(f'{args.parent_dir}/test/{cls}', exist_ok=True)
print("Directories made!")

print("Start downloading files for test data...")
if args.test:
    test_data_slice = test_data[:10]
    print(test_data_slice)
    if args.processes is None:
        download_files(f'{args.parent_dir}/test', test_data_slice)
    else:
        for chunk in chunks(test_data_slice, args.processes):
            download_files(f'{args.parent_dir}/test', chunk)
    exit()
else:
    if args.processes is None:
        download_files(f'{args.parent_dir}/test', test_data)
    else:
        for chunk in chunks(test_data, args.processes):
            download_files(f'{args.parent_dir}/test', chunk)

train_data = read_json(f'json/{args.partition}_train.json')
if args.processes is None:
    download_files(f'{args.parent_dir}/train', train_data)
else:
    for chunk in chunks(train_data, args.processes):
        download_files(f'{args.parent_dir}/train', chunk)
