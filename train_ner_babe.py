from flair.data import Corpus
from flair.datasets import BABE
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.embeddings import CharacterEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


if __name__=='__main__':
    corpus = BABE(base_path='resources/tasks')
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    embedding_types: List[TokenEmbeddings] = [
        # GloVe embeddings
        # WordEmbeddings('glove'),
        CharacterEmbeddings(),
        # contextual string embeddings, forward
        #     PooledFlairEmbeddings('resources/taggers/language_model/best-lm.pt',pooling='mean')
        # contextual string embeddings, backward
        # PooledFlairEmbeddings('news-backward', pooling='min'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger = SequenceTagger(hidden_size=128,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train('resources/taggers/babe-ner',
                  learning_rate=1e-1,
                  train_with_dev=True,
                  monitor_train=True,
                  monitor_test=True,
                  max_epochs=50,
                  use_tensorboard=True)