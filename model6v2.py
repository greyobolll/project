import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

def pad_with_first_row(sequences, maxlen, padding='post', truncating='post'):
    """
    Дополняет последовательности первой строкой вместо нулей.

    :param sequences: Список последовательностей (каждая последовательность — это список списков).
    :param maxlen: Максимальная длина последовательности.
    :param padding: 'post' — дополнение в конце, 'pre' — дополнение в начале.
    :param truncating: 'post' — обрезка в конце, 'pre' — обрезка в начале.
    :return: Дополненный массив NumPy.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Обрезка последовательности, если она длиннее maxlen
            if truncating == 'post':
                seq = seq[:maxlen]
            elif truncating == 'pre':
                seq = seq[-maxlen:]
        else:
            # Дополнение последовательности первой строкой
            first_row = seq[0]  # Первая строка
            padding_length = maxlen - len(seq)
            padding_data = [first_row] * padding_length  # Создаем список из первой строки
            if padding == 'post':
                seq = seq + padding_data
            elif padding == 'pre':
                seq = padding_data + seq
        padded_sequences.append(seq)
    return np.array(padded_sequences)

def read_data_from_file(file_path, has_formula=True):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    X = []
    formula = None

    # Извлекаем данные (игнорируем заголовок и формулу)
    for line in lines[1:-1]:  # Пропускаем первую и последнюю строки
        # Разбиваем строку по пробелам: "0 1 0 = 0" -> ['0', '1', '0', '=', '0']
        parts = line.strip().split()

        # Проверяем корректность формата строки
        if len(parts) < 5 or parts[3] != '=':
            continue  # Пропускаем некорректные строки

        try:
            # Формируем массив из 4 элементов: A, B, C, Res
            a = int(parts[0])
            b = int(parts[1])
            c = int(parts[2])
            res = int(parts[4])
            X.append([a, b, c, res])  # Теперь X содержит 4 параметра
        except (ValueError, IndexError):
            continue  # Пропускаем строки с ошибками

    # Извлекаем формулу (последняя строка файла)
    if has_formula and len(lines) > 1:
        formula_line = lines[-1].strip()
        if '=' not in formula_line:  # Формула не должна содержать '='
            formula = formula_line

    return X, formula

def load_all_data(directory, has_formula=True):
    X_all = []
    formulas = []
    max_length = 9

    files = [f for f in os.listdir(directory) if f.startswith('data_') and f.endswith('.txt')]

    # Формирование данных
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        X, formula = read_data_from_file(file_path, has_formula)
        padded_X = pad_with_first_row([X], maxlen=max_length, padding='post', truncating='post')[0]
        X_all.append(padded_X)
        if formula is not None:
            formulas.append(formula)

    X_all = np.array(X_all)

    # Кодирование формул только для тренировочных данных
    if has_formula:
        unique_formulas = list(set(formulas))
        formula_to_idx = {f: i for i, f in enumerate(unique_formulas)}
        index_to_formula = {i: f for i, f in enumerate(unique_formulas)}
        y = np.array([formula_to_idx[f] for f in formulas])
    else:
        formula_to_idx, index_to_formula, y = None, None, None

    return X_all, y, formula_to_idx, index_to_formula

# Загрузка тренировочных данных
train_directory = r"C:\Users\apmos\PyCharmMiscProject\dataset2\train"
X_train, y_train, formula_to_idx, index_to_formula = load_all_data(train_directory, has_formula=True)

# Проверка данных
print("Уникальные формулы:", index_to_formula.values())
print("Количество формул:", len(index_to_formula))

# Создание модели
model = Sequential([
    Masking(mask_value=0, input_shape=(X_train.shape[1], 4)),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(len(formula_to_idx), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    X_train,
    y_train,
    epochs=400,
    batch_size=32,
    validation_split=0.4
)

# Функция предсказания для тестовых данных
def predict_formula(file_path):
    X, _ = read_data_from_file(file_path, has_formula=False)
    padded_X = pad_with_first_row([X], maxlen=9, padding='post', truncating='post')[0]

    # Добавляем размерность батча (1, 9, 4)
    padded_X = np.expand_dims(padded_X, axis=0)

    probs = model.predict(padded_X)
    formula_idx = np.argmax(probs)
    return index_to_formula[formula_idx]

# Предсказание для тестовых файлов
test_directory = r"C:\Users\apmos\PyCharmMiscProject\dataset2\test"
for i in range(801, 1001):
    file_path = os.path.join(test_directory, f"data_{i}.txt")
    if os.path.exists(file_path):
        predicted_formula = predict_formula(file_path)
        print(f"Файл: data_{i}.txt\tФормула: {predicted_formula}")
    else:
        print(f"Файл data_{i}.txt не найден")
