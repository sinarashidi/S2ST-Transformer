import logging
import pathlib as pl

import numpy as np
import torch
import torchaudio
import tqdm
from torch.nn.parallel import DistributedDataParallel

import speechbrain as sb
from speechbrain.inference.ASR import EncoderASR
from speechbrain.inference.vocoders import UnitHIFIGAN


logger = logging.getLogger(__name__)


class S2ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.src_sig
        tokens_bos, _ = batch.code_bos

        # Use default padding value for wav2vec2
        wavs[wavs == self.hparams.pad_index] = 0.0

        # compute features
        enc_out = self.modules.wav2vec2(wavs, wav_lens)

        # dimensionality reduction
        enc_out = self.modules.enc(enc_out)

        if isinstance(self.modules.transformer, DistributedDataParallel):
            dec_out = self.modules.transformer.module.forward_mt_decoder_only(
                enc_out, tokens_bos, pad_idx=self.hparams.pad_index
            )
        else:
            dec_out = self.modules.transformer.forward_mt_decoder_only(
                enc_out, tokens_bos, pad_idx=self.hparams.pad_index
            )

        # logits and softmax
        pred = self.modules.seq_lin(dec_out)
        p_seq = self.hparams.log_softmax(pred)

        hyps = None
        wavs = None
        transcripts = None
        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.TEST
                or self.hparams.epoch_counter.current
                % self.hparams.evaluation_interval
                == 0
            ):
                ids = batch.id
                tgt_text = batch.tgt_text

                search = (
                    self.hparams.valid_search
                    if stage == sb.Stage.VALID
                    else self.hparams.test_search
                )
                hyps, _, _, _ = search(enc_out.detach(), wav_lens)

                # generate speech and transcriptions
                wavs = []
                for hyp in hyps:
                    if len(hyp) > 3:
                        code = torch.LongTensor(hyp)
                        wav = self.test_vocoder.decode_unit(code)
                        wavs.append(wav.squeeze(0))
                if wavs:
                    wavs, wav_lens = sb.utils.data_utils.batch_pad_right(wavs)
                    transcripts, _ = self.test_asr.transcribe_batch(
                        wavs, wav_lens
                    )
                    transcripts = [
                        transcript.lower() for transcript in transcripts
                    ]

                    self.bleu_metric.append(ids, transcripts, [tgt_text])

        return (
            p_seq,
            wavs,
            transcripts,
        )

    def compute_objectives(self, predictions, batch, stage):
        
        (p_seq, wavs, transcripts) = predictions
        tokens_eos, tokens_eos_lens = batch.code_eos
        ids = batch.id

        # speech translation loss
        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.TEST
                or self.hparams.epoch_counter.current
                % self.hparams.evaluation_interval
                == 0
            ):
                # compute the accuracy of the one-step-forward prediction
                self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

                tgt_wavs, _ = batch.tgt_sig
                tgt_transcripts = batch.tgt_text

                # Save last batch
                wavs = [wav.cpu() for wav in wavs]
                tgt_wavs = [wav.cpu() for wav in tgt_wavs]
                self.last_batch = [
                    ids,
                    (wavs, transcripts),
                    (tgt_transcripts, tgt_wavs),
                ]

        return loss

    def freeze_optimizers(self, optimizers):
        
        valid_optimizers = {}
        if (
            not self.hparams.wav2vec2_frozen
            and self.optimizer_step >= self.hparams.wav2vec2_freeze_steps
        ):
            valid_optimizers["wav2vec_optimizer"] = optimizers[
                "wav2vec_optimizer"
            ]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers

    def init_optimizers(self):

        self.optimizers_dict = {}

        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        self.model_optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict["model_optimizer"] = self.model_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_optimizer", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable(
                "model_optimizer", self.model_optimizer
            )

    def on_fit_batch_start(self, batch, should_step):
        
        if self.optimizer_step == self.hparams.wav2vec2_freeze_steps:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is unfrozen."
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):

        if should_step:
            # anneal model lr every update
            self.hparams.noam_annealing(self.model_optimizer)

    def on_stage_start(self, stage, epoch):

        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.VALID
                and epoch % self.hparams.evaluation_interval != 0
            ):
                return

            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.last_batch = None

            logger.info("Loading pretrained HiFi-GAN ...")
            self.test_vocoder = UnitHIFIGAN.from_hparams(
                source=self.hparams.vocoder_source,
                savedir=self.hparams.vocoder_download_path,
                run_opts={"device": "cpu"},
            )

            logger.info("Loading pretrained ASR ...")
            self.test_asr = EncoderASR.from_hparams(
                source=self.hparams.asr_source,
                savedir=self.hparams.asr_download_path,
                run_opts={"device": "cpu"},
            )

    def on_stage_end(self, stage, stage_loss, epoch):

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        # At the end of validation, we can write
        elif (
            stage == sb.Stage.VALID
            and epoch % self.hparams.evaluation_interval == 0
        ):
            # delete vocoder and asr to free memory for next training epoch
            del self.test_vocoder
            del self.test_asr

            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )

            if output_progress_sample:
                self._save_progress_sample(epoch)

            current_epoch = self.hparams.epoch_counter.current
            lr_model = self.hparams.noam_annealing.current_lr
            lr_wav2vec = 0.0

            if not self.hparams.wav2vec2_frozen:
                (lr_wav2vec, new_lr_wav2vec) = self.hparams.wav2vec_annealing(
                    stage_stats["ACC"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": current_epoch,
                    "lr_model": lr_model,
                    "lr_wav2vec": lr_wav2vec,
                },
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={
                    "ACC": stage_stats["ACC"],
                    "BLEU": stage_stats["BLEU"],
                    "epoch": epoch,
                },
                max_keys=["BLEU"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            stage_stats = {"loss": stage_loss}
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            logger.info(
                f"BLEU score: {round(self.bleu_metric.summarize('BLEU'), 2)}"
            )
            bleu_file = pl.Path(self.hparams.output_folder) / "bleu.txt"
            with open(bleu_file, "a+", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)

    def _save_progress_sample(self, epoch):

        if self.last_batch is None:
            return

        (
            ids,
            (wavs, transcripts),
            (tgt_transcripts, tgt_wavs),
        ) = self.last_batch

        save_folder = pl.Path(self.hparams.progress_sample_path) / f"{epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        sample_size = self.hparams.progress_batch_sample_size
        if len(ids) < sample_size:
            sample_size = len(ids)

        for i in tqdm.tqdm(range(sample_size)):
            utt_id = ids[i]
            wav = wavs[i]
            transcript = transcripts[i]
            tgt_transcript = tgt_transcripts[i]
            tgt_wav = tgt_wavs[i]

            sample_path = save_folder / f"{utt_id}_pred.wav"
            sb.dataio.dataio.write_audio(
                sample_path, wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}_ref.wav"
            sb.dataio.dataio.write_audio(
                sample_path, tgt_wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}.txt"
            with open(sample_path, "w") as file:
                file.write(f"pred: {transcript}\n")
                file.write(f"ref: {tgt_transcript}\n")

        self.bleu_metric.append(
            ids[:sample_size],
            transcripts[:sample_size],
            [tgt_transcripts[:sample_size]],
        )

        bleu_path = save_folder / "bleu.txt"
        with open(bleu_path, "w") as file:
            file.write(
                f"BLEU score: {round(self.bleu_metric.summarize('BLEU'), 2)}\n"
            )


def dataio_prepare(hparams):

    codes_folder = pl.Path(hparams["codes_folder"])

    # Define audio pipeline. In this case, we simply read the audio contained
    # in the variable src_audio with the custom reader.
    @sb.utils.data_pipeline.takes("src_audio")
    @sb.utils.data_pipeline.provides("src_sig")
    def src_audio_pipeline(wav):

        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"]
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("tgt_audio")
    @sb.utils.data_pipeline.provides("tgt_sig")
    def tgt_audio_pipeline(wav):

        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("code_bos", "code_eos")
    def unit_pipeline(utt_id):

        code = np.load(codes_folder / f"{utt_id}_tgt.npy")
        code = torch.LongTensor(code)
        code = torch.unique_consecutive(code)
        code_bos = torch.cat((torch.LongTensor([hparams["bos_index"]]), code))
        yield code_bos
        code_eos = torch.cat((code, torch.LongTensor([hparams["eos_index"]])))
        yield code_eos

    datasets = {}
    for split in hparams["splits"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_json"],
            dynamic_items=[
                src_audio_pipeline,
                tgt_audio_pipeline,
                unit_pipeline,
            ],
            output_keys=[
                "id",
                "src_sig",
                "tgt_sig",
                "duration",
                "code_bos",
                "code_eos",
                "tgt_text",
            ],
        )

    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration"
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration"
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration", reverse=True
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        hparams["valid_dataloader_opts"]["shuffle"] = False

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

    return datasets, train_batch_sampler
