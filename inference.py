import torch

from speechbrain.inference.interfaces import Pretrained


class S2STInference(Pretrained):

    HPARAMS_NEEDED = ["sample_rate"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = self.hparams.sample_rate

    def translate_file(self, path):
        audio = self.load_audio(path)
        audio = audio.to(self.device)
        
        batch = audio.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_tokens = self.translate_batch(batch, rel_length)
        return predicted_tokens[0]

    def encode_batch(self, wavs, wav_lens):
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def translate_batch(self, wavs, wav_lens):
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, _, _, _ = self.mods.decoder(encoder_out, wav_lens)
        return predicted_tokens

    def forward(self, wavs, wav_lens):
        return self.encode_batch(wavs, wav_lens)
