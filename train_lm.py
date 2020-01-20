from flair.data import Dictionary
char_dictionary: Dictionary = Dictionary()
# counter object
import collections
counter = collections.Counter()
processed = 0
import glob
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

if __name__=='__main__':
    files = glob.glob('./corpus/train/*.*')
    max_length = 0
    for file in files:
        print(file)

        with open(file, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                if len(line) > max_length:
                    max_length = len(line)
                processed += 1
                chars = list(line)
                tokens += len(chars)

                # Add chars to the dictionary
                counter.update(chars)

                # comment this line in to speed things up (if the corpus is too large)
                # if tokens > 50000000: break

        # break

    total_count = 0
    for letter, count in counter.most_common():
        total_count += count
    sum = 0
    idx = 0
    for letter, count in counter.most_common():
        sum += count
        percentile = (sum / total_count)

        # comment this line in to use only top X percentile of chars, otherwise filter later
        # if percentile < 0.00001: break

        char_dictionary.add_item(letter)
        idx += 1
        print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

    print(char_dictionary.item2idx)
    is_forward_lm = True
    corpus = TextCorpus('corpus',
                        char_dictionary,
                        is_forward_lm,
                        character_level=True)
    language_model = LanguageModel(char_dictionary,
                                   is_forward_lm,
                                   hidden_size=256,
                                   nlayers=1)

    trainer = LanguageModelTrainer(language_model, corpus)
    trainer.train('resources/taggers/language_model_full',
                  sequence_length=250,
                  mini_batch_size=32,
                  max_epochs=100,
                  checkpoint=False)