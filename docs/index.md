## Description
[XLNet](https://github.com/zihangdai/xlnet) is a generalized autoregressive pretraining method proposed by CMU & Google Brain, which outperforms BERT on 20 NLP tasks ranging from question answering, natural language inference, sentiment analysis, and document ranking. XLNet is inspired by the pros and cons of auto-regressive and auto-encoding methods to overcome limitation of both sides, which uses a permutation language modeling objective to learn bidirectional context and integrates ideas from Transformer-XL into model architecture. This project is aiming to provide extensions built on top of current XLNet and bring power of XLNet to other NLP tasks like NER and NLU.

![xlnet_tasks]({{ site.url }}/xlnet_extension_tf/xlnet.tasks.png){:width="800px"}

*Figure 1: Illustrations of fine-tuning XLNet on different tasks*

## DataSet
* [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/) is a multi-task dataset, which contains 3 sub-tasks, POS tagging, syntactic chunking and NER. For NER sub-task, it contains 4 types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* [ATIS](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html) (Airline Travel Information System) is NLU dataset in airline travel domain. The dataset contains 4978 train and 893 test utterances classified into one of 26 intents, and each token in utterance is labeled with tags from 128 slot filling tags in IOB format.

## Experiment
### CoNLL2003-NER

![xlnet_ner]({{ site.url }}/xlnet_extension_tf/xlnet.ner.png){:width="500px"}

*Figure 2: Illustrations of fine-tuning XLNet on NER task*

|    CoNLL2003 - NER  |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|      Precision      |   91.36 ± 0.50   |     92.14     |
|         Recall      |   92.95 ± 0.24   |     93.20     |
|       F1 Score      |   92.15 ± 0.35   |     92.67     |

*Table 1: The test set performance of XLNet-large finetuned model on CoNLL2003-NER task with setting: batch size = 16, max length = 128, learning rate = 2e-5, num steps = 4,000*

### ATIS-NLU

![xlnet_nlu]({{ site.url }}/xlnet_extension_tf/xlnet.nlu.png){:width="500px"}

*Figure 3: Illustrations of fine-tuning XLNet on NLU task*

|      ATIS - NLU     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|  Accuracy - Intent  |   97.51 ± 0.09   |     97.54     |
|    F1 Score - Slot  |   95.48 ± 0.30   |     95.73     |

*Table 2: The test set performance of XLNet-large finetuned model on ATIS-NLU task with setting: batch size = 16, max length = 128, learning rate = 5e-5, num steps = 2,000*

## Reference
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le and Ruslan Salakhutdinov. [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860) [2019]
* Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee,
and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
