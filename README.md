# UMGTN
The official code of paper, "Unifying Multimodal Source and Propagation Graph for Rumour Detection on Social Media with Missing Features".

## Abstract

With the rapid development of online social media platforms, the spread of rumours has become a critical issue of social concern. Existing methods for rumour detection can be divided into two categories, including image-text pair classification and source-reply graph classification. In this paper, we simultaneously exploit multimodal source and propagation graph features for rumour classification. We propose a Unified Multimodal Graph Transformer Network, namely UMGTN, to fuse the multimodal source and propagation graph features, through Transformer encoders. In social media, not every message is associated with an image. Moreover, the community responses in the propagation graph do not immediately appear after a source message is posted on social media. Therefore, we aim to build a network architecture that accepts inputs with missing features, i.e., missing images or replies. To improve the robustness of the model to data with missing features, we employ a multitask learning framework, by simultaneously learning representations between samples with full and missing features. To evaluate the performance of the proposed method, we extend four real-world datasets, by recovering the images and replies from Twitter and Weibo. Experimental results show that the proposed UMGTN with multitask learning achieves state-of-the-art performance, with a gain of 1.0 to 4.0%, in terms of F1-score, while maintaining the detection robustness to missing features for all datasets within 2%, in terms of accuracy and F1-score, compared to models trained without the multitask learning framework.

## Datasets

Full datasets can be downloaded below. Password will be provided once the paper is accepted.

- [PHEME Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/15083269d_connect_polyu_hk/EWMDKoZhXfNBsjZy0IHwss0B50OhrxctkUWbAiOpq3cuXQ?e=XcTKgo)

- [Twitter Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/15083269d_connect_polyu_hk/EXvz_v8eypRPpa9tGDh79MsB9rPvDrmlNZBwPrh8zHt38w?e=JvNJsl)

- [Weibo-16 Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/15083269d_connect_polyu_hk/EU5duP8NOKtOlvHmpCLp0dEB06TYkwWfhUUyyMEvkFQN1A?e=P8bNqP)

- [Weibo-17 Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/15083269d_connect_polyu_hk/EQfnvtPJPU9DmmyRp1UST8UBjK-MolS9mx2DevZJ8vgoEg?e=SXCxQJ)
