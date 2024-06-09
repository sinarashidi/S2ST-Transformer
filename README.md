# S2ST-Transformer

Direct Speech-to-Speech Translation from Persian language to English using a unit-based Transformer model with a pre-trained Conformer encoder.

🤗 [Model repo on Huggingface](https://huggingface.co/sinarashidi/s2st_fa-en_augmented)

![model](https://github.com/sinarashidi/S2ST-Transformer/assets/97125780/88376e8a-63a0-49af-8972-5b1bad1e58ac)

## Inference
You can use the [Inference Notebook](inference.ipynb) for inference or directly use the inference class implemented in the [Inference File](inference.py).

## Training
Steps for training:
1. Setup environment:
   ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
3. Download the [Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset
4. Download the CVSS dataset
5. Change the ```src_data_folder``` and ```tgt_data_folder``` in the [hyperparams.yaml](hyperparams.yaml) file according to your setup
6. Change other training parameters or model configs in the [hyperparams.yaml](hyperparams.yaml) file if needed
7. Start the training procedure:
   ```bash
     python train.py hyperparams.yaml --precision=bf16
With default settings, each epoch takes about 2 hours on the RTX 3090 GPU. 

To run only the evaluation stage on a checkpoint, run the following command:
```bash
  python train.py hyperparams.yaml --precision=bf16 --test_only
