# Continual Learning on Distilled Dataset

This repository contains the code for __Section 3.7 Application: Continual Learning__ of the paper ["Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective"](https://arxiv.org/abs/2306.13092). [[SRe<sup>2</sup>L Project Page]](https://zeyuanyin.github.io/projects/SRe2L/)

## Requirements

- Python 3.8
- PyTorch 1.13.1
- torchvision 0.14.1
- numpy
- scipy
- tqdm

## Usage

Firstly, follow [SRe2L](https://github.com/VILA-Lab/SRe2L) to get the squeezed model and distilled Tiny-ImageNet dataset (100 IPC). Then, run the following [script](run.sh) to run continual learning on the distilled dataset.

```bash
python main.py  \
  --steps 5 --lr_net 0.5 \
  -T 20 --num_eval 3 --ipc 100 \
  --train_dir /path/to/distilled_tiny \
  --teacher_path /path/to/tiny-imagenet/resnet18_E50/checkpoint.pth \
  | tee  cl_sre2l_T20_step5.txt
```

You can find the example output at [cl_sre2l_T20_step5.txt](cl_sre2l_T20_step5.txt).

## Citation

If you find our code useful for your research, please cite our paper.

```bibtex
@inproceedings{yin2023squeeze,
  title={Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective},
  author={Yin, Zeyuan and Xing, Eric and Shen, Zhiqiang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgement

This repository is built upon the codebase of <https://github.com/VICO-UoE/DatasetCondensation>. We thank the authors for their great work.
