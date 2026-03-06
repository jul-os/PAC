import codecs
import numpy as np
import gensim

def one_hot_encode_fragments(fragments, chunk_size=None):
    if chunk_size is None:
        chunk_size = len(fragments[0]) if fragments else 0
    
    vocab = sorted(set(''.join(fragments)))
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)

    n_fragments = len(fragments)
    encoded = np.zeros((n_fragments, chunk_size, vocab_size), dtype=np.float32)
    
    for frag_idx, fragment in enumerate(fragments):
        for char_idx, char in enumerate(fragment):
            if char in char_to_idx:
                encoded[frag_idx, char_idx, char_to_idx[char]] = 1.0
    
    return encoded, char_to_idx


def split_text_equal_parts(text, chunk_size):
    text = text.strip()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


with codecs.open('//home/julia/Рабочий стол/code/PAC/RNN/измененные андроиды.txt', encoding='utf-8', mode='r') as f:    
    docs = f.readlines()
    
max_sentence_len = 12

sentences = [sent for doc in docs for sent in doc.split('.')]
sentences = [[word for word in sent.lower().split()[:max_sentence_len]] for sent in sentences]
print(len(sentences), 'предложений')

# Обучение модели
word_model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, window=5, epochs=100)

pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape
print(vocab_size, embedding_size)

print('Похожие слова:')
for word in ['андроид', 'рик', 'декард', 'тест', 'сова', 'паук']:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
    print('  %s -> %s' % (word, most_similar))


file_path = '/home/julia/Рабочий стол/code/PAC/RNN/fragment.txt'

chunk_size = 100

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

result = split_text_equal_parts(content, chunk_size)
encoded, char_to_idx = one_hot_encode_fragments(result, chunk_size)


print(f"\nФорма результата: {encoded.shape}")

"""
На encoded обучить модель RNN для предсказания следующего символа. Посмотрите результат при последовательной генерации.
"""