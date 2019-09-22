import math
import os
import tensorflow as tf
import urllib.request

# load Tensorflow/Google Brain base code
# https://github.com/tensorflow/models/tree/master/research/im2txt

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary


# tell our function where to find the trained model and vocabulary
checkpoint_path = './model'
vocab_file = './model/word_counts.txt'

# this is the function we'll call to produce our captions 
#    given input file name(s) -- separate file names by a ,
#                                 if more than one

def gen_caption(path):
    with urllib.request.urlopen(path) as url:
        with open('temp/temp.jpg', 'wb') as f:
            f.write(url.read())
    input_files = 'temp/temp.jpg'

    # only print serious log messages
    tf.logging.set_verbosity(tf.logging.FATAL)
    # load our pretrained model
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                 checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(vocab_file)

    filenames = []
    for file_pattern in input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), input_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        
        captionlist = []

        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                captionlist.append(sentence)
    return {"captions": captionlist}
