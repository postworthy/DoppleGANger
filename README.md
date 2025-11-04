# DoppleGANger

DoppleGANger is a TensorFlow-based face swapping pipeline that wraps an AEI-GAN style generator, discriminator, and optional super-resolution upsampler. The project provides training and inference utilities tuned for CUDA-enabled environments to produce high fidelity swaps on still images and video.

## Environment

- Copy `sample.env` to `.env` and fill in dataset and model cache paths.
- Run `./run.sh` to rebuild the CUDA container and drop into the project shell before installing dependencies, training, or running inference.

## Inference

Inside the container:

```bash
python src/gradio_app.py --model /models/MODEL_256x256_SUPER_v14_BLOCKS2_latest
```

The Gradio UI wraps `AEINETSwapper` for quick smoke tests on images or videos. Adjust the `--model` flag if you have a different checkpoint under `models/`. For low-level access, use `src/inference.py` directly.

## Training

Still inside the container, you can kick off the standard training loop with:

```bash
bash src/train_model.sh
```

This script forwards to `src/train.py` with defaults that fine-tune the super-resolution path. To customize hyperparameters, modify the shell script or call the trainer directly, e.g.:

```bash
python src/train.py --help
```

Training expects TFRecord shards under the path provided via `--tfrecord_shard_path` and will save checkpoints inside `models/`. Tune learning rates, block counts, and dataset locations through CLI arguments as needed.
