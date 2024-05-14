python main.py  \
  --steps 5 --lr_net 0.5 \
  -T 20 --num_eval 3 --ipc 100 \
  --train_dir /path/to/distilled_tiny \
  --teacher_path /path/to/tiny-imagenet/resnet18_E50/checkpoint.pth \
  | tee  cl_sre2l_T20_step5.txt