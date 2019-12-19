# XLNet Extension
[XLNet](https://github.com/zihangdai/xlnet) is a generalized autoregressive pretraining method proposed by CMU & Google Brain, which outperforms BERT on 20 NLP tasks ranging from question answering, natural language inference, sentiment analysis, and document ranking. XLNet is inspired by the pros and cons of auto-regressive and auto-encoding methods to overcome limitation of both sides, which uses a permutation language modeling objective to learn bidirectional context and integrates ideas from Transformer-XL into model architecture. This project is aiming to provide extensions built on top of current XLNet and bring power of XLNet to other NLP tasks like NER and NLU.
<p align="center"><img src="/docs/xlnet.tasks.png" width=800></p>
<p align="center"><i>Figure 1: Illustrations of fine-tuning XLNet on different tasks</i></p>

## Setting
* Python 3.6.7
* Tensorflow 1.13.1
* NumPy 1.13.3
* SentencePiece 0.1.82

## DataSet
* [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/) is a multi-task dataset, which contains 3 sub-tasks, POS tagging, syntactic chunking and NER. For NER sub-task, it contains 4 types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* [ATIS](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html) (Airline Travel Information System) is NLU dataset in airline travel domain. The dataset contains 4978 train and 893 test utterances classified into one of 26 intents, and each token in utterance is labeled with tags from 128 slot filling tags in IOB format.
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
* [CoQA](https://stanfordnlp.github.io/coqa/) a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation. CoQA is pronounced as coca
* [QuAC](https://quac.ai/) is a dataset for modeling, understanding, and participating in information seeking dialog. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context.

## Usage
* Preprocess data
```bash
python prepro/prepro_conll.py \
  --data_format json \
  --input_file data/ner/conll2003/raw/eng.xxx \
  --output_file data/ner/conll2003/xxx-conll2003/xxx-conll2003.json
```
* Run experiment
```bash
CUDA_VISIBLE_DEVICES=0 python run_ner.py \
    --spiece_model_file=model/cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=model/cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=model/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --task_name=conll2003 \
    --random_seed=100 \
    --predict_tag=xxxxx \
    --data_dir=data/ner/conll2003 \
    --output_dir=output/ner/conll2003/data \
    --model_dir=output/ner/conll2003/checkpoint \
    --export_dir=output/ner/conll2003/export \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=2e-5 \
    --train_steps=2500 \
    --warmup_steps=100 \
    --save_steps=500 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --do_export=true
```
* Visualize summary
```bash
tensorboard --logdir=output/ner/conll2003
```
* Setup service
```bash
docker run -p 8500:8500 \
  -v output/ner/conll2003/export/xxxxx:models/ner \
  -e MODEL_NAME=ner \
  -t tensorflow/serving
```

## Experiment
### CoNLL2003-NER
<p align="center"><img src="/docs/xlnet.ner.png" width=500></p>
<p align="center"><i>Figure 2: Illustrations of fine-tuning XLNet on CoNLL2003-NER task</i></p>

|    CoNLL2003 - NER  |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|      Precision      |   91.36 ± 0.50   |     92.14     |
|         Recall      |   92.95 ± 0.24   |     93.20     |
|       F1 Score      |   92.15 ± 0.35   |     92.67     |

<p><i>Table 1: The test set performance of XLNet-large finetuned model on CoNLL2003-NER task with setting: batch size = 16, max length = 128, learning rate = 2e-5, num steps = 4,000</i></p>

### ATIS-NLU
<p align="center"><img src="/docs/xlnet.nlu.png" width=500></p>
<p align="center"><i>Figure 3: Illustrations of fine-tuning XLNet on ATIS-NLU task</i></p>

|      ATIS - NLU     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|  Accuracy - Intent  |   97.51 ± 0.09   |     97.54     |
|    F1 Score - Slot  |   95.48 ± 0.30   |     95.73     |

<p><i>Table 2: The test set performance of XLNet-large finetuned model on ATIS-NLU task with setting: batch size = 16, max length = 128, learning rate = 5e-5, num steps = 2,000</i></p>

### SQuAD v1.1
<p align="center"><img src="/docs/xlnet.squad.v1.png" width=500></p>
<p align="center"><i>Figure 4: Illustrations of fine-tuning XLNet on SQuAD v1.1 task</i></p>

|      SQuAD v1.1     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|     Exact Match     |   xx.xx ± x.xx   |     88.61     |
|       F1 Score      |   xx.xx ± x.xx   |     94.28     |

<p><i>Table 3: The test set performance of XLNet-large finetuned model on SQuAD v1.1 task with setting: batch size = 48, max sequence length = 512, max question length = 64, learning rate = 3e-5, num steps = 8,000</i></p>

### SQuAD v2.0
<p align="center"><img src="/docs/xlnet.squad.v2.png" width=500></p>
<p align="center"><i>Figure 5: Illustrations of fine-tuning XLNet on SQuAD v2.0 task</i></p>

|      SQuAD v2.0     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|     Exact Match     |   xx.xx ± x.xx   |     85.72     |
|       F1 Score      |   xx.xx ± x.xx   |     88.36     |

<p><i>Table 4: The test set performance of XLNet-large finetuned model on SQuAD v2.0 task with setting: batch size = 48, max sequence length = 512, max question length = 64, learning rate = 3e-5, num steps = 8,000</i></p>

### CoQA v1.0
<p align="center"><img src="/docs/xlnet.coqa.png" width=500></p>
<p align="center"><i>Figure 6: Illustrations of fine-tuning XLNet on CoQA v1.0 task</i></p>

|       CoQA v1.0     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|     Exact Match     |   xx.xx ± x.xx   |     xx.xx     |
|       F1 Score      |   xx.xx ± x.xx   |     xx.xx     |

<p><i>Table 5: The test set performance of XLNet-large finetuned model on CoQA v1.0 task with setting: batch size = 48, max sequence length = 512, max question length = 128, learning rate = 3e-5, num steps = 6,000</i></p>

### QuAC v0.2
<p align="center"><img src="/docs/xlnet.quac.png" width=500></p>
<p align="center"><i>Figure 7: Illustrations of fine-tuning XLNet on QuAC v0.2 task</i></p>

|       QuAC v0.2     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|       F1 Score      |   xx.xx ± x.xx   |     xx.xx     |
|         HEQQ        |   xx.xx ± x.xx   |     xx.xx     |
|         HEQD        |   xx.xx ± x.xx   |     xx.xx     |

<p><i>Table 6: The test set performance of XLNet-large finetuned model on QuAC v0.2 task with setting: batch size = 48, max sequence length = 512, max question length = 128, learning rate = 2e-5, num steps = 8,000</i></p>

## Reference
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. [SQuAD: 100,000+ questions for machine comprehension of text](https://arxiv.org/abs/1606.05250) [2016]
* Pranav Rajpurkar, Robin Jia, and Percy Liang. [Know what you don’t know: unanswerable questions for SQuAD](https://arxiv.org/abs/1806.03822) [2018]
* Siva Reddy, Danqi Chen, Christopher D. Manning. [CoQA: A Conversational Question Answering Challenge](https://arxiv.org/abs/1808.07042) [2018]
* Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, Luke Zettlemoyer. [QuAC : Question Answering in Context](https://arxiv.org/abs/1808.07036) [2018]
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee, and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
* Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) [2019]
* Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) [2019]
* Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) [2019]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le and Ruslan Salakhutdinov. [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860) [2019]
