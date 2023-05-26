"""
@author : quvox, junkurihara
@when : 2023-5-26
@homepage : https://github.com/junkurihara/transformer
"""
from typing import Iterable, List, Tuple
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import torchtext.transforms as T


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
BATCH_SIZE = 128

class DataLoader:
    def __init__(self, ext, init_token, eos_token):
        self.ext = ext
        self.tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')
        self.tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        train_iter, val_iter, test_iter = Multi30k(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en')) # type: ignore
        return train_iter, val_iter, test_iter

    # helper function to yield list of tokens
    def yield_tokens_de(self, data_iter: Iterable) -> List[str]: # type: ignore
        for data_sample in data_iter:
            yield self.tokenize_de(data_sample[0])

    # helper function to yield list of tokens
    def yield_tokens_en(self, data_iter: Iterable) -> List[str]: # type: ignore
        for data_sample in data_iter:
            yield self.tokenize_en(data_sample[0])

    def build_vocab(self, data_iter: Iterable):
        # Create torchtext's Vocab object
        vocab_transform_de = build_vocab_from_iterator(self.yield_tokens_de(data_iter),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform_en = build_vocab_from_iterator(self.yield_tokens_en(data_iter),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
        vocab_transform_de.set_default_index(UNK_IDX)
        vocab_transform_en.set_default_index(UNK_IDX)

        self.source_vocab = vocab_transform_de
        self.target_vocab = vocab_transform_en

    # helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input) # type: ignore
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))


    # def into_tensor(self, batch):
        # def text_transform_de(b):
        #     b = self.tokenize_de(b)
        #     b = self.tensor_transform(b)
        #     return b
        # text_transform_de = self.sequential_transforms(
        #     self.tokenize_de, #Tokenization
        #                                               # self.source_vocab, #Numericalization # type: ignore
        #                                               # self.tensor_transform
        #                                               ) # Add BOS/EOS and create tensor  # type: ignore
        # text_transform_en = self.sequential_transforms(
        #     self.tokenize_en, #Tokenization
        #                                               # self.target_vocab, #Numericalization # type: ignore
        #                                               # self.tensor_transform
        #                                               ) # Add BOS/EOS and create tensor  # type: ignore

        # src_batch, src_len, tgt_batch = [], [], []
        # for src_sample, tgt_sample in batch:
        #     src_batch.append(text_transform_de(src_sample.rstrip("\n"))) # type: ignore
        #     src_len.append(len(src_batch[-1]))
        #     tgt_batch.append(text_transform_en(tgt_sample.rstrip("\n"))) # type: ignore

        #     src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        #     tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

        # return src_batch, torch.LongTensor(src_len), tgt_batch

    def collate_fn(self, batch):
        src_batch, src_len, tgt_batch = [], [], []
        for src_sample, tgt_sample in batch:
            # Process source
            tokenized_src = self.tokenize_de(src_sample.rstrip("\n"))
            numerized_src = self.source_vocab(tokenized_src)
            src_batch.append(self.tensor_transform(numerized_src)) # type: ignore
            src_len.append(len(src_batch[-1]))

            # Process target
            tokenized_tgt = self.tokenize_en(tgt_sample.rstrip("\n"))
            numerized_tgt = self.target_vocab(tokenized_tgt)
            tgt_batch.append(self.tensor_transform(numerized_tgt)) # type: ignore

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

        return src_batch, torch.LongTensor(src_len), tgt_batch

    def get_dataloader(self, train_iter, val_iter, test_iter):
        self.train_dataloader = TorchDataLoader(list(train_iter), batch_size=BATCH_SIZE, collate_fn=self.collate_fn) # type: ignore
        self.val_dataloader = TorchDataLoader(list(val_iter), batch_size=BATCH_SIZE, collate_fn=self.collate_fn) # type: ignore
        self.test_dataloader = TorchDataLoader(list(test_iter), batch_size=BATCH_SIZE, collate_fn=self.collate_fn) # type: ignore
        return self.train_dataloader, self.val_dataloader, self.test_dataloader


if __name__ == "__main__":
    import copy
    loader = DataLoader(ext=('.en', '.de'),
                        init_token='<sos>',
                        eos_token='<eos>')
    train_iter, val_iter, test_iter = loader.make_dataset()
    loader.build_vocab(copy.deepcopy(train_iter))

    train, val, test = loader.get_dataloader(train_iter, val_iter, test_iter)

    print(loader)
    for i, batch in enumerate(val):
        print(i, batch)
