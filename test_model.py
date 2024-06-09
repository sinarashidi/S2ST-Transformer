import torch
import torchaudio
from inference import S2STInference
from speechbrain.inference.vocoders import UnitHIFIGAN
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Translate speech using S2STInference and UnitHIFIGAN.")
parser.add_argument('input_file', type=str, help='Path to the input audio file')
parser.add_argument('output_file', type=str, help='Path to save the output translated audio file')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

s2ut = S2STInference.from_hparams(source="sinarashidi/s2st_fa-en_augmented", savedir="tmpdir_s2ut", run_opts={'device': device})
hifi_gan_unit = UnitHIFIGAN.from_hparams(source="speechbrain/tts-hifigan-unit-hubert-l6-k100-ljspeech", savedir="tmpdir_vocoder", run_opts={'device': device})

file = args.input_file
output_file = args.output_file

codes = s2ut.translate_file(file)
codes = torch.IntTensor(codes)

waveforms = hifi_gan_unit.decode_unit(codes)
torchaudio.save(output_file, waveforms.squeeze(1).cpu(), 16000)
