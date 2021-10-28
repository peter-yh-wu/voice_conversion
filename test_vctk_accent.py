import random
import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import pickle
from utils import Hps
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
from scipy.io.wavfile import write
import os
import argparse
from solver import Solver


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps', help='The path of hyper-parameter set', default='vctk.json')
    parser.add_argument('-model', '-m', help='The path of model checkpoint')
    parser.add_argument('-target', '-t', help='Target speaker id (integer). Same order as the speaker list when preprocessing (en_speaker_used.txt)')
    parser.add_argument('-output', '-o', help='output .wav path')
    parser.add_argument('-data', default='VCTK-Corpus')
    parser.add_argument('--use_gen', default=True, action='store_true')
    parser.add_argument('--mode', help='default, test, train, train[num]', default='default', type=str)
    parser.add_argument('--tag', help='tag', default='', type=str)
    parser.add_argument('--seed', help='seed', default=0, type=int)
    args = parser.parse_args()

    data_dir = args.data
    metadata_path = os.path.join(data_dir, 'speaker-info.txt')
    wav_dir = os.path.join(data_dir, 'wav48')
    if args.tag == '':
        new_wav_dir = os.path.join(data_dir, 'wav_gvc_accent')
    else:
        new_wav_dir = os.path.join(data_dir, 'wav_gvc_accent_%s' % args.tag)
    if not os.path.exists(new_wav_dir):
        os.makedirs(new_wav_dir)
    with open(metadata_path, 'r') as inf:
        metadata = inf.readlines()
    # 225  23  F    English    Southern  England
    metadata = [l.strip() for l in metadata][1:] # ignore header line
    metadata = [l for l in metadata if len(l) > 0 and l[0] != ';']
    metadata = [l.split(' ') for l in metadata]
    metadata = [[l.strip() for l in ll if l.strip() != ''] for ll in metadata]
    id_to_gender = {l[0]:l[2] for l in metadata}
    id_to_accent = {l[0]:l[3] for l in metadata}
    accents = sorted(list(set(id_to_accent.values())))

    train_speaker_txt_path = 'train_speaker_ids.txt'
    with open(train_speaker_txt_path, 'r') as inf:
        speakers = inf.readlines()
    train_speakers = [l.strip() for l in speakers]
    train_speakers_set = set(train_speakers)

    speaker_txt_path = 'test_speaker_ids.txt'
    with open(speaker_txt_path, 'r') as inf:
        speakers = inf.readlines()
    test_speakers = [l.strip() for l in speakers]
    test_speakers_set = set(test_speakers)

    if args.mode == 'default':
        all_speakers = train_speakers + test_speakers
    elif args.mode == 'test':
        all_speakers = test_speakers
    elif args.mode == 'train':
        all_speakers = train_speakers
    elif 'train' in args.mode:
        speaker_txt_path = train_speaker_txt_path.replace('train', args.mode+'-%d' % args.seed)
        if not os.path.exists(speaker_txt_path):
            random.Random(0).shuffle(train_speakers)
            num_speakers = int(args.mode[5:])
            all_speakers = train_speakers[:num_speakers]
            with open(speaker_txt_path, 'w+') as ouf:
                for s in all_speakers:
                    ouf.write('%s\n' % s)
        else:
            with open(speaker_txt_path, 'r') as inf:
                speakers = inf.readlines()
            all_speakers = [l.strip() for l in speakers]
    else:
        print('unsupported mode')
        exit()

    with open('vctk_gvc_speaker_mappings.txt', 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    s_to_ci = {}
    for l in lines:
        l_list = l.split()
        s = l_list[0]
        cs = l_list[1] # chosen speaker
        ci = l_list[2] # chosen speaker idx
        s_to_ci[s] = ci

    all_wav_paths = []
    all_cis = []
    for s in all_speakers:
        speaker_dir = os.path.join(wav_dir, 'p'+s)
        speaker_files = os.listdir(speaker_dir)
        speaker_files = [f for f in speaker_files if f.endswith('.wav')]
        wav_paths = [os.path.join(speaker_dir, f) for f in speaker_files]
        all_wav_paths += wav_paths
        ci = s_to_ci[s]
        all_cis += [ci]*len(wav_paths)

    new_dirs = set()
    for cpath in all_wav_paths:
        relpath = os.path.relpath(cpath, wav_dir)
        cdir = os.path.dirname(relpath)
        new_dir = os.path.join(new_wav_dir, cdir)
        new_dirs.add(new_dir)
    
    new_dirs = list(new_dirs)
    for d in new_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    print(len(all_wav_paths))

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
        for (path, ci) in tqdm(zip(all_wav_paths, all_cis), total=len(all_wav_paths)):
            relpath = os.path.relpath(path, wav_dir)
            output_path = os.path.join(new_wav_dir, relpath)
            _, spec = get_spectrograms(path)
            spec_expand = np.expand_dims(spec, axis=0)
            spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
            c = torch.from_numpy(np.array([int(ci)])).cuda(0)
            result = solver.test_step(spec_tensor, c, gen=args.use_gen)
            result = result.squeeze(axis=0).transpose((1, 0))
            wav_data = spectrogram2wav(result)
            write(output_path, rate=16000, data=wav_data)
