import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from model import S2ST, dataio_prepare


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from cvss_prepare import prepare_cvss

    sb.utils.distributed.run_on_main(
        prepare_cvss,
        kwargs={
            "src_data_folder": hparams["src_data_folder"],
            "tgt_data_folder": hparams["tgt_data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    from extract_code import extract_cvss

    sb.utils.distributed.run_on_main(
        extract_cvss,
        kwargs={
            "data_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "kmeans_folder": hparams["kmeans_source"],
            "encoder": hparams["encoder_source"],
            "layer": hparams["layer"],
            "save_folder": hparams["save_folder"],
            "sample_rate": hparams["sample_rate"],
            "skip_extract": hparams["skip_extract"],
        },
    )

    datasets, train_bsampler = dataio_prepare(hparams)

    s2st_brain = S2ST(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
            "collate_fn": hparams["train_dataloader_opts"]["collate_fn"],
        }

    s2st_brain.fit(
        s2st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid_small"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    test_dataloader_opts = {
        "batch_size": 1,
    }

    for dataset in ["valid", "test"]:
        s2st_brain.evaluate(
            datasets[dataset],
            max_key="BLEU",
            test_loader_kwargs=test_dataloader_opts,
        )
