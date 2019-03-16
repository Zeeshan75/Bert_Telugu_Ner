from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
import pickle

# para = 'సభికుడు నిజమే కానీ ఖంగారు పడద్దు అని భార్యను ఇలా ఓదార్చాడు – “మూర్ఖుల మనసులో ఒకటి పడితే అది సాధించాలన్న పట్టుదల బలంగా మొదలవుతుంది. వారితో వాదించడం కష్టం. మహారాజు గారి మనసులో ఈ విషయం అలాగే బలంగా పడిపోయింది. వారిని కాదన్న వారి తలలు నరికించేసారూ. నన్ను అడిగిన వెంటనే నేను కూడా కుదరదు అంటే నా తల కూడా వెంటనే తెగేది.'
# para = 'అప్పుడు శవంలోని బేతాళుడు, “రాజా, విద్యాదికుదివైన నువ్వు, భీతగోల్పే ఈ స్మశానంలో, నిశిరాత్రి వేళ ఇన్ని కష్టాలను సాహిస్తున్నావంటే నమ్మసఖ్యం కాకుండా వున్నది. ఇంతకూ దీనికి కారకులు, నీ మంచితనాన్ని తమ స్వార్థం కోసం ఉపయోగించు కుంటున్న కుత్సిత మనస్కులై వుండాలి. అలాంటి వాళ్ళ వలలో చిక్కి, కార్యసాధన తర్వాత అవివేకం కొద్దీ తనమేలును కూడా మరచిన అనటుడనే యువకుడి కథ చెబుతాను, శ్రమ తెలియకుండా విను.” అంటూ ఇలా చెప్పసాగాడు:'
# para = 'విరజుడికి యక్షిణి శైలం గురించి తెలుసు. మందార దేశానికి ఉత్తరపు టెల్లగావున్న పెద్ద పర్వతాన్ని యక్షిణి శైలం గా వర్ణించి చెబుతారు. ఆ పర్వతం మీదవున్న ప్రాచీన శివాలయం ముఖమండపం దగ్గర, దేవనాగారలిపిలో వున్న ఒక శిలా శాసనం వుంది. అందులో, ఈ ఆలయం ముందున్న కోనేటి మధ్యభాగంలో నిరంతరం వేగంగా తిరిగేసుడి ఒకటున్నది. మంత్రం తంత్ర శాస్త్రజ్ఞాని, సాహసీ, నిస్వార్థ పరుడూ అయిన యువకుడు, ఆ సుడిలో ప్రవేశించి అక్కడ ఎదురయ్యే యక్షమాయను జయిన్చినట్లయితే, అతడికి అష్ట సిద్ధులూ, నవనిదులూ లభిస్తాయి, పరాజితుడైతే అక్కడి నుంచి నరక కూపంలోకి తోయ బాదుతాడు, అని వున్నది. సుప్రతీకుడి మాటలో అదంతా గుర్తు తెచ్చుకుని విరజుడు బిక్కముఖం వేసి, “యక్షిణీ శైలం గురించి తెలిసి ఏం లాభం? ప్రాణాలకు తెగించి, ఆ కోనేటిలో దూకడం నా వల్ల కాదు!” అన్నాడు.'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            print('**************************',input_file)
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0 and words[-1] == '.':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    
                    s = l.split()
                    for i in range(len(s)):
                        if s[i].endswith('\u200c'):
                            s[i]= s[i][:-1]
                        else:
                            continue
                    li = ' '.join(s)
                    
                    s = w.split()
                    for i in range(len(s)):
                        if s[i].endswith('\u200c'):
                            s[i]= s[i][:-1]
                        else:
                            continue
                    wo = ' '.join(s)
                    
                    lines.append([li, wo])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            # print(lines)
            # print(len(lines))
            return lines
class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")


    def get_labels(self):
        return ["B-MISC", "I-MISC", "O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode):
    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with open('./output/label2id.pkl','wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, max_seq_length, 13])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits,predict)
        ##########################################################################
        
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,  per_example_loss,logits,predicts) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def metric_fn(per_example_loss, label_ids, logits):
            # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,13,[1,2,4,5,6,7,8,9],average="macro")
                recall = tf_metrics.recall(label_ids,predictions,13,[1,2,4,5,6,7,8,9],average="macro")
                f = tf_metrics.f1(label_ids,predictions,13,[1,2,4,5,6,7,8,9],average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions= predicts,scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn

data_dir = None
bert_config_file = None  #   "The config json file corresponding to the pre-trained BERT model."
task_name = None        #    "The name of the task to train."
output_dir = None       #    "The output directory where the model checkpoints will be written."
init_checkpoint = None  #    "Initial checkpoint (usually from a pre-trained BERT model)."
do_lower_case = True    #    "Whether to lower case the input text."
max_seq_length = 128    #    "The maximum total input sequence length after WordPiece tokenization."
do_train = False        #    "Whether to run training."
use_tpu = False         #    "Whether to use TPU or GPU/CPU.")

do_eval = False         # "Whether to run eval on the dev set.")
do_predict = False      # "Whether to run the model in inference mode on the test set.")
train_batch_size = 32   #  "Total batch size for training.")
eval_batch_size = 8     #  "Total batch size for eval.")
predict_batch_size = 8  #  "Total batch size for predict.")
learning_rate = 5e-5    #  "The initial learning rate for Adam.")
num_train_epochs = 3.0  #  "Total number of training epochs to perform.")
warmup_proportion = 0.1 #    "Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10% of training.")
iterations_per_loop = 1000   # "How many steps to make in each estimator call."
vocab_file = None       # "The vocabulary file that the BERT model was trained on.")
master = None           # "[Optional] TensorFlow master URL.")
num_tpu_cores = 8       # "Only used if `use_tpu` is True. Total number of TPU cores to use.")

from tensorflow.contrib.cluster_resolver import TPUClusterResolver

try:
	files_list = [i.split('.')[0] for i in os.listdir('models/multi_cased_L-12_H-768_A-12')]
except:
	print("Please create a directory 'models/multi_cased_L-12_H-768_A-12'.")
	exit();
files = ['vocab.txt','bert_config.json','bert_model.ckpt']
dict1 = {}

for file in files:
	if file.split('.')[0] in files_list:
		file_name = file.split('.')[0]+'_file'
		path = 'models/multi_cased_L-12_H-768_A-12/'+file
		dict1[file_name] = path
	else:
		print(file," not present in 'models/multi_cased_L-12_H-768_A-12'.")
		exit();


# vocab_file = 'bert/models/multi_cased_L-12_H-768_A-12/vocab.txt'
# bert_config_file = 'bert/models/multi_cased_L-12_H-768_A-12/bert_config.json'
# init_checkpoint = 'bert/models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'

vocab_file = dict1['vocab_file']
bert_config_file = dict1['bert_config_file']
init_checkpoint = dict1['bert_model_file']

data_dir = 'input/'
output_dir = 'output/result_dir/'
task_name = 'NER'
do_train = False  
do_eval = False   
do_predict = True 
max_seq_length = 128
train_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 3.0
save_checkpoints_steps = 150
    
tf.logging.set_verbosity(tf.logging.INFO)
processors = {
    "ner": NerProcessor
}
# if not do_train and not do_eval:
#     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

if max_seq_length > bert_config.max_position_embeddings:
	raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

task_name = task_name.lower()
if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))
processor = processors[task_name]()

label_list = processor.get_labels()

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case = do_lower_case)
tpu_cluster_resolver = None
tpu_name = 'grpc://10.54.242.34:8470'
if use_tpu and tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_name)#, project=gcp_project) #zone=tpu_zone, 

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=master,
    model_dir=output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=num_tpu_cores,
        per_host_input_for_training=is_per_host))

train_examples = None
num_train_steps = None
num_warmup_steps = None

if do_train:
    train_examples = processor.get_train_examples(data_dir)
    num_train_steps = int(
        len(train_examples) / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list)+1,
    init_checkpoint=init_checkpoint,
    learning_rate=learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=use_tpu,
    use_one_hot_embeddings=use_tpu)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    predict_batch_size=predict_batch_size)


def predict(para):
	if do_predict:
		# para = "There's a passage I got memorized. Ezekiel 25:17. The path of the righteous man is beset on all sides by the inequities of the selfish and the tyranny of evil men."
		output_predict_file =  'input/test.txt'
		with open(output_predict_file,'w') as writer:
			writer.write('-DOCSTART-\n \n')
			for lines in para.split('.'):
				if len(lines)>1:
					words = lines.split()
					words.append('.')
					for i in words:
						if '\u200c' in i:
							i=i.replace('\u200c','')
						writer.write(i+' X\n')
						if i == '.':
							writer.write(' \n')


		token_path = os.path.join(output_dir, "token_test.txt")
		with open('./output/label2id.pkl','rb') as rf:
			label2id = pickle.load(rf)
			id2label = {value:key for key,value in label2id.items()}
		if os.path.exists(token_path):
			os.remove(token_path)
		predict_examples = processor.get_test_examples(data_dir)
		predict_file = os.path.join(output_dir, "predict.tf_record")
		filed_based_convert_examples_to_features(predict_examples, label_list, max_seq_length, tokenizer, predict_file,mode="test")
		tf.logging.info("***** Running prediction*****")
		tf.logging.info("  Num examples = %d", len(predict_examples))
		tf.logging.info("  Batch size = %d", predict_batch_size)
		predict_drop_remainder = True if use_tpu else False
		predict_input_fn = file_based_input_fn_builder(
			input_file=predict_file,
			seq_length=max_seq_length,
			is_training=False,
			drop_remainder=predict_drop_remainder)
		# disp = []
		result = estimator.predict(input_fn=predict_input_fn)
		output_predict_file = os.path.join('.', "output_file.txt")
		with open(output_predict_file,'w') as writer:
			for prediction in result:
				output_line = "\n".join(id2label[id] for id in prediction if id!=0) + "\n"
				writer.write(output_line)
				# disp.append(output_line)
	output_file = 'output_file.txt'
	lo = []
	to = []
	with open(output_file) as f:
		for line in f:
			line = line[:-1]
			if line.strip() == '[CLS]':
				continue
			elif line.strip() == '[SEP]':
				continue
			else:
				if line.strip() == 'X':
					continue
				else:
					lo.append(line)

	input_file = 'input/test.txt'
	l1= []
	with open(input_file) as f:
		for line in f:
			line = line[:-1]
			if line.strip() == '-DOCSTART-' or line.strip() == '':
				continue
			else:
				line = line.split()[0]  #('X',lo[i])
			l1.append(line)
	rest_response = { 'status' : True, 'message' : 'Successfully Predicted the Named Entities.'}
	result = dict(zip(l1,lo))
	print(result)
	rest_response['result'] = str(result)
	return rest_response

    # for generating the new output file.
	# result = dict(zip(l1,lo))
	# print(result)
	# fout = "new_output.txt"
	# fo = open(fout, "w")

	# for k, v in result.items():
	# 	fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
	# fo.close()

if __name__ == '__main__':
	a= "'కాకులు ఒక పొలానికి వెళ్లి భరత్ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి. పిచుక నిస్సహాయంగా ఏమి చేయాలో ఆదివారం తెలీకా అటూ ఇటూ గెంతుతూ వుంది. పదమూడేళ్ల ఇంతలో ఆ పొలం రైతులు పరిగెత్తుకుంటూ వచ్చి ఒక పెద్ద కర్రతో\u200c ఆ కాకులను కొట్టడం మొదలెట్టారు. కాకుల\u200c గుంపుకు ఇది అలవాటే, అవి తుర్రున ఎగిరిపోయాయి. పిచుక రైతులకు దొరికిపోయింది.'"
	print(predict(a))