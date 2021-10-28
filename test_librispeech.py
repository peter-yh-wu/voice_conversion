import glob
import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import pickle
from utils import Hps
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
from scipy.io.wavfile import write
import glob
import os
import argparse
from solver import Solver

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps', help='The path of hyper-parameter set', default='vctk.json')
    parser.add_argument('-model', '-m', help='The path of model checkpoint')
    parser.add_argument('-source', '-s', help='The path of source .wav file')
    parser.add_argument('-target', '-t', help='Target speaker id (integer). Same order as the speaker list when preprocessing (en_speaker_used.txt)')
    parser.add_argument('-output', '-o', help='output .wav path')
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    parser.add_argument('-data', default='train-clean-100')
    parser.add_argument('--use_gen', default=True, action='store_true')

    args = parser.parse_args()

    paths = []
    ext = 'flac'

    data_dir = args.data
    search_path = os.path.join(data_dir, '**/*.' + ext)
    for fname in glob.iglob(search_path, recursive=True):
        file_path = os.path.realpath(fname)
        paths.append(file_path)
    paths = [p for p in paths if '-avc' not in p]
    paths = [p for p in paths if '-gvc' not in p]
    speaker_ids = []
    for file_path in paths:
        slash_idx = file_path.rfind('/')
        end_idx = file_path[:slash_idx].rfind('/')
        start_idx = file_path[:end_idx].rfind('/')+1
        speaker_id = file_path[start_idx:end_idx]
        speaker_ids.append(speaker_id)
    speakers = ['311', '2843', '3664', '3168', '2518', '7190', '78', '831', '8630', '3830', '322', '2391', '7517', '8324', '19', '1898', '7078', '5339', '4051', '4640']
    speaker_set = set(speakers)
    paths = [p for i, p in enumerate(paths) if speaker_ids[i] in speaker_set]

    hps = Hps()
    hps.load(args.hps)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    
    num_params1 = count_parameters(solver.Encoder)
    num_params2 = count_parameters(solver.Decoder)
    num_params3 = count_parameters(solver.Generator)
    print(num_params1+num_params2+num_params3)
    
    solver.load_model(args.model)

    with torch.no_grad():
        for path in tqdm(paths):
            output_path = path.replace('.flac', '-gvc.flac')
            _, spec = get_spectrograms(path)
            spec_expand = np.expand_dims(spec, axis=0)
            spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
            c = torch.from_numpy(np.array([int(args.target)])).cuda(1)
            result = solver.test_step(spec_tensor, c, gen=args.use_gen)
            result = result.squeeze(axis=0).transpose((1, 0))
            wav_data = spectrogram2wav(result)
            write(output_path, rate=args.sample_rate, data=wav_data)
