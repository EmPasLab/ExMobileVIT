# ExMobileViT: Lightweight Classifier Extension for Mobile Vision Transformer

> Abstract
> The paper proposes an efficient structure for enhancing the performance of mobile-friendly vision transformer with small computational overhead. The vision transformer (ViT) is very attractive in that it reaches outperforming results in image classification, compared to conventional convolutional neural networks (CNNs). Due to its need of high computational resources, MobileNet-based ViT models such as MobileViT-S have been developed. However, their performance cannot reach the original ViT model. The proposed structure relieves the above weakness by storing the information from early attention stages and reusing it in the final classifier. This paper is motivated by the idea that the data itself from early attention stages can have important meaning for the final classification. In order to reuse the early information in attention stages, the average pooling results of various scaled features from early attention stages are used to expand channels in the fully-connected layer of the final classifier. It is expected that the inductive bias introduced by the averaged features can enhance the final performance. Because the proposed structure only needs the average pooling of features from the attention stages and channel expansions in the final classifier, its computational and storage overheads are very small, keeping the benefits of low-cost MobileNet-based ViT (MobileViT). Compared with the original MobileViTs on ImageNet dataset, the proposed ExMobileViT has noticeable accuracy enhancements, having only about 5% additional parameters.

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
