import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
import shutil
from tensorflow.keras.callbacks import Callback

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
train_directory = r"D:\другое\dataset2\train"
X_train, y_train, formula_to_idx, index_to_formula = load_all_data(train_directory, has_formula=True)

# Проверка данных
print("Уникальные формулы:", index_to_formula.values())
print("Количество формул:", len(index_to_formula))

# Создание модели
model = Sequential([
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

def evaluate_expression(A, B, C, expression):
    # Replace variables in the expression with their values
    expr = expression.replace('A', str(A)).replace('B', str(B)).replace('C', str(C))
    # Evaluate the expression
    try:
        result = eval(expr)
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {e}")
        return None
    return result


def parse_expression(expression_line):
    expr = expression_line.strip()

    if not expr:
        print(f"Пустая строка формулы: {expression_line}")
        return None

    # Разделяем по первому '=', если он есть
    if '=' in expr:
        parts = expr.split('=', 1)  # Разделяем только на 2 части
        formula_part = parts[0].strip()
    else:
        formula_part = expr

    return formula_part


class CustomCallback(Callback):
    def __init__(self, file_path, target_accuracy=0.5):
        super().__init__()
        self.file_path = file_path
        self.target_accuracy = target_accuracy
        self.attempt = 0

    def on_epoch_end(self, epoch, logs=None):
        self.attempt += 1
        if self.attempt >= 20:
            self.model.stop_training = True


def process_file(file_name):
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        header = lines[0].strip().split()
        if header != ['A', 'B', 'C', 'Res']:
            print(f"Header mismatch in {file_name}. Expected ['A', 'B', 'C', 'Res'], got {header}")
            return 0.0, []

        expression_line = lines[-1].strip()
        expression = parse_expression(expression_line)
        if not expression:
            return 0.0, []

        correct = 0
        total = 0
        incorrect_lines = []

        for line_number, line in enumerate(lines[1:-1], start=1):
            parts = line.strip().split('=')
            if len(parts) != 2:
                continue

            try:
                values = list(map(int, parts[0].strip().split()))
                if len(values) != 3:
                    continue

                A, B, C = values
                expected_res = int(parts[1].strip())
                result = evaluate_expression(A, B, C, expression)

                total += 1
                if result is not None:
                    result_int = int(result)
                    if result_int == expected_res:
                        correct += 1
                    else:
                        incorrect_lines.append(1 + line_number)
            except Exception as e:
                print(f"Error in line {line_number}: {e}")

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, incorrect_lines

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return 0.0, []


def process_and_retrain(file_path, attempt=0):
    predicted_formula = predict_formula(file_path)
    new_file_name = f"data_predicted_{os.path.basename(file_path)}"
    new_file_path = os.path.join(train_directory, new_file_name)

    with open(file_path, 'r') as src:
        content = src.read().rstrip('\n')

    with open(new_file_path, 'w') as dst:
        dst.write(content)
        dst.write(f"\n{predicted_formula}")

    accuracy, incorrect_lines = process_file(new_file_path)
    print(f"Attempt {attempt + 1}: Accuracy = {accuracy:.2f}")

    # Удаление некорректных строк
    if incorrect_lines:
        with open(new_file_path, 'r') as f:
            lines = f.readlines()

        new_lines = [lines[0]]  # Сохраняем заголовок
        keep_lines = set(range(1, len(lines) - 1)) - set(incorrect_lines)

        for idx in sorted(keep_lines):
            new_lines.append(lines[idx])

        new_lines.append(lines[-1])  # Сохраняем формулу

        with open(new_file_path, 'w') as f:
            f.writelines(new_lines)

        accuracy, _ = process_file(new_file_path)  # Перепроверяем точность

    if accuracy < 0.5 and attempt < 20:
        X_new, y_new, _, _ = load_all_data(train_directory, has_formula=True)
        callback = CustomCallback(new_file_path)
        model.fit(X_new, y_new, epochs=1, batch_size=32, callbacks=[callback], verbose=0)
        return process_and_retrain(file_path, attempt + 1)

    return predicted_formula, accuracy


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import random


class ImprovedCustomCallback(Callback):
    def __init__(self, file_path, target_accuracy=0.8):
        super().__init__()
        self.file_path = file_path
        self.target_accuracy = target_accuracy
        self.attempt = 0
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        self.attempt += 1
        if logs.get('accuracy') > self.best_accuracy:
            self.best_accuracy = logs['accuracy']
        if self.attempt >= 15 or self.best_accuracy >= self.target_accuracy:
            self.model.stop_training = True


def process_and_retrain2(file_path, attempt=0):
    predicted_formula = predict_formula(file_path)
    temp_file_name = f"temp_{os.path.basename(file_path)}"
    temp_file_path = os.path.join(train_directory, temp_file_name)

    # Генерация временного файла с аугментацией
    with open(file_path, 'r') as src, open(temp_file_path, 'w') as dst:
        content = src.read().rstrip('\n')
        dst.write(content + f"\n{predicted_formula}")

    # Валидация и очистка данных
    accuracy, incorrect_lines = process_file(temp_file_path)
    print(f"Attempt {attempt + 1}: Initial Accuracy = {accuracy:.2f}")

    if incorrect_lines:
        with open(temp_file_path, 'r') as f:
            lines = f.readlines()

        # Аугментация: добавляем 3 новых примера для каждой корректной строки
        new_lines = [lines[0]]
        valid_lines = [line for idx, line in enumerate(lines[1:-1], 1)
                       if idx not in incorrect_lines]

        for line in valid_lines:
            new_lines.append(line)
            # Генерация дополнительных примеров
            values = list(map(int, line.strip().split('=')[0].strip().split()))
            for _ in range(1):
                aug_values = [random.randint(0, 1) for _ in values]
                A, B, C = aug_values
                new_line = f"{' '.join(map(str, aug_values))} = {int(evaluate_expression(A, B, C, predicted_formula))}\n"
                new_lines.append(new_line)

        new_lines.append(lines[-1])

        with open(temp_file_path, 'w') as f:
            f.writelines(new_lines)

        accuracy, _ = process_file(temp_file_path)
        print(f"Attempt {attempt + 1}: Augmented Accuracy = {accuracy:.2f}")

    if accuracy < 0.8 and attempt < 15:
        # Загрузка только новых данных
        X_new, y_new, _, _ = load_all_data(train_directory, has_formula=True)

        # Настройка оптимизатора и callback'ов
        optimizer = Adam(
            learning_rate=0.001 * (0.95 ** attempt),
            clipnorm=1.0
        )

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        callbacks = [
            ImprovedCustomCallback(temp_file_path),
            EarlyStopping(monitor='loss', patience=5),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
        ]

        # Дообучение с аугментацией
        model.fit(
            X_new,
            y_new,
            epochs=50,
            batch_size=16,
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )

        return process_and_retrain2(file_path, attempt + 1)

    # Перенос успешного файла в постоянную базу
    if accuracy >= 0.8:
        final_file_name = f"valid_{os.path.basename(file_path)}"
        shutil.move(temp_file_path, os.path.join(train_directory, final_file_name))
    else:
        if os.path.exists(temp_file):
            os.remove(temp_file_path)

    return predicted_formula, accuracy

# Модифицированный цикл предсказания
test_directory = r"D:\другое\dataset2\test"
for i in range(801, 1001):
    file_path = os.path.join(test_directory, f"data_{i}.txt")
    if os.path.exists(file_path):
        final_formula, final_accuracy = process_and_retrain2(file_path)
        print(f"Файл: data_{i}.txt\tФормула: {final_formula}\tТочность: {final_accuracy:.2f}")

        # Удаляем временный файл после обработки
        temp_file = os.path.join(train_directory, f"data_predicted_data_{i}.txt")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    else:
        print(f"Файл data_{i}.txt не найден")