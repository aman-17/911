with open("gpt_practise/tokenization/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print(len(raw_text))


import re
# x = re.split(r'([,.-?:;()"\'_=+[]{}<>`~]|--|\s)', raw_text)
x = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
x_new = [item for item in x if item.split()]
# print(x_new[:30])

all_words = sorted(list(set(x_new)))
# vocab_size = len(all_words)
# print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
# print(vocab)
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i>30:
#         break


def encode(text):
    x = re.split(r'([,.?_!"()\']|--|\s)', text)
    x_new = [item for item in x if item.split()]
    all_words = sorted(list(set(x_new)))
    vocab = {token:integer for integer,token in enumerate(all_words)}
    return vocab

encodee = encode(raw_text)


def decode(encodee, vocab):
    int_to_str = {i: s for s, i in vocab.items()}
    text = " ".join([int_to_str[i] for i in encodee.values()])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text

print(decode(encodee=encodee, vocab=vocab))


# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab #A
#         self.int_to_str = {i:s for s,i in vocab.items()} #B

#     def encode(self, text): #C
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [item if item in self.str_to_int  #A
#                         else "<|unk|>" for item in preprocessed]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
    
#     def decode(self, ids): #D
#         text = " ".join([self.int_to_str[i] for i in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
#         return text
    

# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know," Mrs. Gisburn said pard"""
# ids = tokenizer.encode(text)
# print(ids)