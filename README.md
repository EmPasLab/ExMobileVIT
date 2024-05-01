# ExMobileViT: Lightweight Classifier Extension for Mobile Vision Transformer

- This code is utlizing basic technique. If you want to run code, utilze below code.
```commandline
python main.py
```

- Script have parsers to run code. These are examples of utlizing parser.
```commandline
python main.py --data_root ~/database/imagenet --batch_size 128 --log_dir ~/log/ExMobileViT --log_name ExMobileViT-928
```

- Ubuntu-based machine with 80 threads2 from 2 Intel Xeon Gold 5218R CPUs, 6 Nvidia RTX A5000 GPUs, 256 Gigabytes main memory, and 4 days are used for training.

- ExMobileViT model is written in models directory: ExMobileViT_576, ExMobileViT_640, ExMobileViT_928. Also baseline model, MobileViT is in models directory.
- If you want to adjust model, make model file in models directory and change making_model file.

- To check parser file, find in utils directory.