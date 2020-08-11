import glob
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
from pathlib import Path
from typing import Tuple


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--target-path', '-tp', help='directly point to target dataset path (overrides hparams.target_wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path
if args.target_path is None:
    args.target_path = hp.target_wav_path

extension = args.extension
path = args.path
target_path = args.target_path


def convert_files(path: Path, path_target: Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak

    y_target = load_wav(path_target)
    peak = np.abs(y_target).max()
    if hp.peak_norm or peak > 1.0:
        y_target /= peak

    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y_target, mu=2**hp.bits) if hp.mu_law else float_2_label(y_target, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y_target, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wavs(wav_paths: Tuple[Path, Path]):
    path, path_target = wav_paths
    wav_id = path.stem
    m, x = convert_files(path, path_target)
    np.save(paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]


wav_files = sorted(get_files(path, extension))
target_wav_files = sorted(get_files(target_path, extension))
paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')
print(f'\n{len(target_wav_files)} {extension[1:]} target files found in "{target_path}"\n')

if len(wav_files) == 0 or len(target_wav_files) == 0:

    print('Please point wav_path / target_wav_path in hparams.py to your dataset,')
    print('or use the --path / --target-path option.\n')

else:

    if not hp.ignore_tts:

        text_dict = ljspeech(path)

        with open(paths.data/'text_dict.pkl', 'wb') as f:
            pickle.dump(text_dict, f)

    n_workers = max(1, args.num_workers)

    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Mu Law', hp.mu_law),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}')
    ])

    pool = Pool(processes=n_workers)
    dataset = []

    for i, (item_id, length) in enumerate(pool.imap_unordered(process_wavs, list(zip(wav_files, target_wav_files))), 1):
        dataset += [(item_id, length)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    with open(paths.data/'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
