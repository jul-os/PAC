import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

# ============================================================================
# ПУНКТ 1: WORD2VEC
# ============================================================================
print("=" * 60)
print("ПУНКТ 1: WORD2VEC")
print("=" * 60)

file_path = 'измененные андроиды.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"✓ Текст загружен: {len(raw_text)} символов")
except FileNotFoundError:
    print(f"⚠️ Файл не найден")

# Очистка текста (только буквы и пробелы)
def clean_text(text):
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

cleaned_text = clean_text(raw_text)
words = cleaned_text.split()

# Word2Vec
try:
    from gensim.models import Word2Vec
    
    sentences = [words[i:i+10] for i in range(0, len(words), 10)]
    
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )
    
    print(f"✓ Word2Vec обучена (слов: {len(w2v_model.wv)})")
    
    test_words = ['андроид', 'человек', 'мечта']
    print("\n🔍 Похожие слова:")
    for word in test_words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=3)
            print(f"  '{word}' → {', '.join([w for w, _ in similar])}")
            
except ImportError:
    print("⚠️ Gensim не установлен (pip install gensim)")
    w2v_model = None

# ============================================================================
# ПУНКТ 2: ПОДГОТОВКА ДАННЫХ (СИМВОЛЫ + ONE-HOT)
# ============================================================================
print("\n" + "=" * 60)
print("ПУНКТ 2: ПОДГОТОВКА ДАННЫХ (ONE-HOT)")
print("=" * 60)

# 🔥 УВЕЛИЧЬТЕ ЭТО ЗНАЧЕНИЕ для лучшего качества (50000-100000)
FRAGMENT_LENGTH = 10000  
SENTENCE_LEN = 50

text_fragment = cleaned_text[:FRAGMENT_LENGTH]

def split_into_fixed_length(text, seq_length):
    sequences = []
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
    return sequences

sequences = split_into_fixed_length(text_fragment, SENTENCE_LEN)
print(f"Количество последовательностей: {len(sequences)}")

char_to_idx = {ch: idx for idx, ch in enumerate(sorted(list(set(text_fragment))))}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
vocab_size = len(char_to_idx)

print(f"Размер алфавита: {vocab_size}")
print(f"Форма данных: (n={len(sequences)}, len={SENTENCE_LEN}, enc={vocab_size})")

def one_hot_encode(char, char_to_idx, vocab_size):
    vector = np.zeros(vocab_size)
    vector[char_to_idx.get(char, 0)] = 1
    return vector

def encode_sequence(seq, char_to_idx, vocab_size):
    return np.array([one_hot_encode(ch, char_to_idx, vocab_size) for ch in seq])

X_data = []
y_data = []

for i, seq in enumerate(sequences):
    X_data.append(encode_sequence(seq, char_to_idx, vocab_size))
    next_idx = i + SENTENCE_LEN
    if next_idx < len(text_fragment):
        next_char = text_fragment[next_idx]
    else:
        next_char = seq[-1]
    y_data.append(one_hot_encode(next_char, char_to_idx, vocab_size))

X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

print(f"✓ Форма X: {X_data.shape}")
print(f"✓ Форма Y: {y_data.shape}")

# ============================================================================
# ПУНКТ 3: RNN + ОБУЧЕНИЕ + ГЕНЕРАЦИЯ
# ============================================================================
print("\n" + "=" * 60)
print("ПУНКТ 3: RNN (С УЧЕТОМ HIDDEN STATE)")
print("=" * 60)

class CharDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
# 🔥 УВЕЛИЧЬТЕ для лучшего качества (50-100)
EPOCHS = 30

dataset = CharDataset(X_data, y_data)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

model = CharRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- ОБУЧЕНИЕ ---
print("\nОбучение...")
print("-" * 60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output, _ = model(x_batch, hidden=None)
        
        y_indices = y_batch.argmax(dim=1)
        loss = criterion(output, y_indices)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f}")

print("\n✓ Обучение завершено")

# ============================================================================
# 🔥 ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ (ОДИН СИМВОЛ + DETACH)
# ============================================================================
print("\n" + "=" * 60)
print("ГЕНЕРАЦИЯ (ИСПРАВЛЕННАЯ)")
print("=" * 60)

def generate_text_char_correct(model, seed_text, char_to_idx, idx_to_char, 
                               vocab_size, seq_length=50, gen_length=100, 
                               temperature=0.8):
    model.eval()
    seed_text = clean_text(seed_text)
    
    if len(seed_text) < seq_length:
        seed_text = ' ' * (seq_length - len(seed_text)) + seed_text
    
    hidden = None
    
    with torch.no_grad():
        # 🔥 ШАГ 1: ПРОГРЕВ (заполняем hidden state затравкой)
        for i in range(len(seed_text)):
            char = seed_text[i]
            input_vec = one_hot_encode(char, char_to_idx, vocab_size)
            # shape: (batch=1, seq_len=1, vocab_size)
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).unsqueeze(0).to(device)
            _, hidden = model(input_tensor, hidden)
            
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
        
        # 🔥 ШАГ 2: ГЕНЕРАЦИЯ (по одному символу)
        generated = ""
        current_char = seed_text[-1]
        
        for _ in range(gen_length):
            input_vec = one_hot_encode(current_char, char_to_idx, vocab_size)
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).unsqueeze(0).to(device)
            
            output, hidden = model(input_tensor, hidden)
            
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            # 🔥 ИСПРАВЛЕНИЕ ОШИБКИ: .detach() перед .numpy()
            if temperature > 0:
                probs = torch.softmax(output / temperature, dim=1).detach().cpu().numpy()[0]
                next_idx = np.random.choice(len(probs), p=probs)
            else:
                next_idx = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
            
            next_char = idx_to_char.get(next_idx, ' ')
            generated += next_char
            current_char = next_char
    
    return generated

# Тестирование
seeds = ["андроид", "человек", "время", "мечта"]

print("\nРезультаты генерации:")
print("-" * 60)

for seed in seeds:
    print(f"\n🔹 Затравка: '{seed}'")
    for temp in [0.5, 0.8]:
        text = generate_text_char_correct(model, seed, char_to_idx, idx_to_char,
                                          vocab_size, seq_length=SENTENCE_LEN,
                                          gen_length=150, temperature=temp)
        print(f"  temp={temp}: {text[:100]}...")

print("\n" + "=" * 60)
print("✓ ЗАДАНИЕ ВЫПОЛНЕНО")
print("=" * 60)
    #сохранять контекст при 
    #hidden отбрасывать, не сохранять между батчами при тренировке
    #в eval режиме hidden использовать при генерации следующего символа
