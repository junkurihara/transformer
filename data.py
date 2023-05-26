"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
# from util.data_loader import DataLoader
# from util.tokenizer import Tokenizer
from new_data_loader import DataLoader
import copy

# tokenizer = Tokenizer()
# loader = DataLoader(ext=('.en', '.de'),
#                     tokenize_en=tokenizer.tokenize_en,
#                     tokenize_de=tokenizer.tokenize_de,
#                     init_token='<sos>',
#                     eos_token='<eos>')

# train, valid, test = loader.make_dataset()
# loader.build_vocab(train_data=train, min_freq=2)
# train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
#                                                      batch_size=batch_size,
#                                                      device=device)
loader = DataLoader(ext=('.en', '.de'),
                        init_token='<bos>',
                        eos_token='<eos>')
train_iter, valid_iter, test_iter = loader.make_dataset()
loader.build_vocab(copy.deepcopy(train_iter))
loader.get_dataloader(train_iter, valid_iter, test_iter)

src_pad_idx = loader.source_vocab.get_stoi()['<pad>']
trg_pad_idx = loader.target_vocab.get_stoi()['<pad>']
trg_sos_idx = loader.target_vocab.get_stoi()['<bos>']

enc_voc_size = len(loader.source_vocab)
dec_voc_size = len(loader.target_vocab)
