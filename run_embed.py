from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import sentencepiece as sp

from xlnet import xlnet
from xlnet import prepro_utils
from xlnet import model_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "Data directory where raw data located.")
flags.DEFINE_string("output_dir", None, "Output directory where processed data located.")
flags.DEFINE_string("model_dir", None, "Model directory where checkpoints located.")
flags.DEFINE_string("export_dir", None, "Export directory where saved model located.")

flags.DEFINE_string("model_config_path", None, "Config file of the pre-trained model.")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint of the pre-trained model.")
flags.DEFINE_string("model_type", "sent", "Model type: e.g. 'sent', 'token', etc.")

flags.DEFINE_bool("lower_case", False, "Enable lower case nor not.")
flags.DEFINE_string("spiece_model_file", None, "Sentence Piece model path.")
flags.DEFINE_integer("max_seq_length", 128, "Max sequence length")

flags.DEFINE_float("dropout", 0.1, "Dropout rate.")
flags.DEFINE_float("dropatt", 0.1, "Attention dropout rate.")
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", 0.02, "Initialization std when init is normal.")
flags.DEFINE_float("init_range", 0.1, "Initialization std when init is uniform.")
flags.DEFINE_integer("clamp_len", -1, "Clamp length")
flags.DEFINE_bool("use_bfloat16", False, "Whether to use bfloat16.")

flags.DEFINE_integer("train_steps", 1, "Number of training steps")
flags.DEFINE_integer("warmup_steps", 0, "number of warmup steps")
flags.DEFINE_float("learning_rate", 1e-5, "initial learning rate")
flags.DEFINE_float("min_lr_ratio", 0.0, "min lr ratio for cos decay.")
flags.DEFINE_float("clip", 1.0, "Gradient clipping")
flags.DEFINE_float("weight_decay", 0.0, "Weight decay rate")
flags.DEFINE_float("adam_epsilon", 1e-8, "Adam epsilon")
flags.DEFINE_string("decay_method", "poly", "poly or cos")
flags.DEFINE_integer("max_save", 1, "Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", 1, "Save the model for every save_steps. If None, not to save any model.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("num_hosts", 1, "How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", 1, "Total number of TPU cores to use.")
flags.DEFINE_string("tpu_job_name", None, "TPU worker job name.")
flags.DEFINE_string("tpu", None, "The Cloud TPU name to use for training.")
flags.DEFINE_string("tpu_zone", None, "GCE zone where the Cloud TPU is located in.")
flags.DEFINE_string("gcp_project", None, "Project name for the Cloud TPU-enabled project.")
flags.DEFINE_string("master", None, "TensorFlow master URL")
flags.DEFINE_integer("iterations", 1, "number of iterations per TPU training loop.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""
    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case
    
    def tokenize(self,
                 text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text, return_unicode=False)
        return tokenized_pieces
    
    def encode(self,
               text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids
    
    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)
    
    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)
    
    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]

class XLNetExampleConverter(object):
    """Default example converter for XLNet"""
    def __init__(self,
                 label_list,
                 max_seq_length,
                 tokenizer):
        """Construct XLNet example converter"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<a>", "<b>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i
        
        self.label_list = label_list
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i
        
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
    
    def _truncate_seq_pair(self,
                           tokens_a,
                           tokens_b,
                           max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    
    def convert_single_example(self,
                               example,
                               logging=False):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * self.max_seq_length,
                input_mask=[1] * self.max_seq_length,
                segment_ids=[0] * self.max_seq_length,
                label_id=0)
        
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        # The convention in XLNet is:
        # (a) For sequence pairs:
        #  tokens:      is it a dog ? [SEP] no , it is not . [SEP] [CLS] 
        #  segment_ids: 0  0  0 0   0 0     1  1 1  1  1   1 1     2
        # (b) For single sequences:
        #  tokens:      this dog is big . [SEP] [CLS] 
        #  segment_ids: 0    0   0  0   0 0     2
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the last vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense when
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(self.segment_vocab_map["<a>"])

        tokens.append("<sep>")
        segment_ids.append(self.segment_vocab_map["<a>"])

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(self.segment_vocab_map["<b>"])

            tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<b>"])
        
        tokens.append("<cls>")
        segment_ids.append(self.segment_vocab_map["<cls>"])
        
        input_ids = tokenizer.tokens_to_ids(tokens)

        # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        if len(input_ids) < self.max_seq_length:
            pad_seq_length = self.max_seq_length - len(input_ids)
            input_ids = [self.special_vocab_map["<pad>"]] * pad_seq_length + input_ids
            input_mask = [1] * pad_seq_length + input_mask
            segment_ids = [self.segment_vocab_map["<pad>"]] * pad_seq_length + segment_ids
        
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = label_map[example.label]
        if logging:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join([printable_text(token) for token in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
        
        return feature
    
    def convert_examples_to_features(self,
                                     examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

            feature = self.convert_single_example(example, logging=(idx < 5))
            features.append(feature)

        return features
    
    def file_based_convert_examples_to_features(self,
                                                examples,
                                                output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        
        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for (idx, example) in enumerate(examples):
                if idx % 10000 == 0:
                    tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

                feature = convert_single_example(example, logging=(idx < 5))

                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_float_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature([feature.label_id])

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

class XLNetInputBuilder(object):
    """Default input builder for XLNet"""
    @staticmethod
    def get_input_builder(features,
                          is_training,
                          seq_length,
                          drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)
        
        def input_fn(params,
                     input_context=None):
            batch_size = params["batch_size"]
            num_examples = len(features)
            
            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.float32),
                "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "label_ids": tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
            })
            
            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id, input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            
            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d
        
        return input_fn
    
    @staticmethod
    def get_file_based_input_fn(input_file,
                                is_training,
                                seq_length,
                                drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }
        
        def _decode_record(record,
                           name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example
        
        def input_fn(params,
                     input_context=None):
            """The actual input function."""
            batch_size = params["batch_size"]
            
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            
            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id, input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
            
            return d
        
        return input_fn
    
    @staticmethod
    def get_serving_input_fn(seq_length):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        def serving_input_fn():
            with tf.variable_scope("serving"):
                features = {
                    'input_ids': tf.placeholder(tf.int32, [None, seq_length], name='input_ids'),
                    'input_mask': tf.placeholder(tf.float32, [None, seq_length], name='input_mask'),
                    'segment_ids': tf.placeholder(tf.int32, [None, seq_length], name='segment_ids')
                }

                return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()
        
        return serving_input_fn

class XLNetModelBuilder(object):
    """Default model builder for XLNet"""
    def __init__(self,
                 default_model_config,
                 default_run_config,
                 default_init_checkpoint,
                 use_tpu=False):
        """Construct XLNet model builder"""
        self.default_model_config = default_model_config
        self.default_run_config = default_run_config
        self.default_init_checkpoint = default_init_checkpoint
        self.use_tpu = use_tpu
    
    def _create_model(self,
                      model_config,
                      run_config,
                      input_ids,
                      input_mask,
                      segment_ids,
                      model_type):
        """Create XLNet core model"""
        model = xlnet.XLNetModel(
            xlnet_config=model_config,
            run_config=run_config,
            input_ids=input_ids,
            input_mask=input_mask,
            seg_ids=segment_ids)
        
        if model_type == "token":
            output_result = model.get_sequence_output()
        else:
            output_result = model.get_pooled_out("last")
        
        return output_result
    
    def get_model_fn(self,
                     model_config,
                     run_config,
                     init_checkpoint,
                     model_type):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features,
                     labels,
                     mode,
                     params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]

            embeddings = self._create_model(model_config, run_config, input_ids, input_mask, segment_ids, model_type)
            scaffold_fn = model_utils.init_from_checkpoint(FLAGS)
            
            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                loss = tf.Variable(0.0, name="loss", dtype=tf.float32)
                train_op, _, _ = model_utils.get_train_op(FLAGS, loss)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={ "embeddings": embeddings },
                    scaffold_fn=scaffold_fn)
            
            return output_spec
        
        return model_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    tpu_config = model_utils.configure_tpu(FLAGS)
    model_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(False, True, FLAGS)
    
    model_builder = XLNetModelBuilder(
        default_model_config=model_config,
        default_run_config=run_config,
        default_init_checkpoint=FLAGS.init_checkpoint,
        use_tpu=FLAGS.use_tpu)
    
    model_fn = model_builder.get_model_fn(model_config, run_config, FLAGS.init_checkpoint, FLAGS.model_type)
    
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=tpu_config,
        export_to_tpu=FLAGS.use_tpu,
        train_batch_size=1)
    
    tokenizer = XLNetTokenizer(
        sp_model_file=FLAGS.spiece_model_file,
        lower_case=FLAGS.lower_case)
    
    example_converter = XLNetExampleConverter(
        label_list=[],
        max_seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer)
    
    features = example_converter.convert_examples_to_features([PaddingInputExample()])
    
    train_input_fn = XLNetInputBuilder.get_input_builder(features, True, FLAGS.max_seq_length, False)
    estimator.train(train_input_fn, max_steps=1)
    
    tf.gfile.MakeDirs(FLAGS.export_dir)
    serving_input_fn = XLNetInputBuilder.get_serving_input_fn(FLAGS.max_seq_length)
    estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn, as_text=False)

if __name__ == "__main__":
    flags.mark_flag_as_required("spiece_model_file")
    flags.mark_flag_as_required("model_config_path")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("export_dir")
    tf.app.run()
