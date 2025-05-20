import re
import tqdm
import json

class TokenizerV0:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def load_txt_file(raw_text) -> dict:
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item.strip()]
    all_words = sorted(list(set(preprocessed)))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_words)}
    return vocab


# Iterable dataset still pending
def stream_json_data(file_path, chunk_size=100*1024*1024):
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""
        for chunk in iter(lambda: f.read(chunk_size), ''):
            buffer += chunk
            try:
                last_bracket = buffer.rindex('}')
                valid_json = buffer[:last_bracket+1]
                data = json.loads(valid_json)
                yield data
                buffer = buffer[last_bracket+1:]
            except (ValueError, json.JSONDecodeError):
                continue

def build_vocab_streaming(file_path, vocab_size=50000):
    word_counts = {}
    for data_chunk in tqdm(stream_json_data(file_path)):
        text = data_chunk.get("text", "")
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [t for t in tokens if t.strip()]
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, _ in sorted_words[:vocab_size-2]]
    top_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: i for i, token in enumerate(top_words)}
    return vocab
    
