# tom-steer

![alt text](image.png)

This is a course project for CSE 598-012 Science of LLMs (Winter 2025) at the University of Michigan - Ann Arbor. Team members: Yuchen Huang, Tongyuan Miao, and Tianshuo Yang.

## Abstract

Understanding the internal representations of large language models (LLMs) is critical for enhancing their reasoning capabilities. Sparse autoencoders (SAEs) have recently emerged as a promising tool for uncovering interpretable, monosemantic features from model activations. In this project, we investigate whether SAEs trained on GPT-2 Small encode aspects of social reasoning, specifically Theory of Mind (ToM) capabilities. We construct two controlled datasets—a ToM-relevant dataset derived from ToMi and a baseline dataset from bAbI—and analyze the divergence of SAE activations between them. By identifying features with the highest activation differences and manually curating their semantic descriptions via Neuronpedia's Autointerp, we select candidate features for activation steering. Steering experiments show that selectively amplifying certain features can modestly improve model performance on ToM tasks, suggesting that SAEs capture latent cognitive dimensions beyond lexical semantics. Our findings demonstrate the potential of SAE-based feature steering as a lightweight post-training technique for enhancing specific reasoning abilities in language models, while also highlighting challenges related to model scale, dataset bias, and feature interpretability.

