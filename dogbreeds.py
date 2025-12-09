
# 
import os
import pandas as pd
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 120 # количество пород собак в датасете
DATA_DIR = 'archive.zip/'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
MODEL_SAVE_PATH = 'dog_breed_classifier_mobilenetv2.h5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# генерация (Dataset для PyTorch)
class DogBreedDataGenerator(Dataset):
    def __init__(self, dataframe, img_size, num_classes, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row.filepath
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row.encoded_labels)
        return img, label

def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def predict_breed(image_path, model, label_encoder, img_size):
    try:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)  # shape [1,3,H,W]

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))

        predicted_breed = label_encoder.inverse_transform([predicted_class_idx])[0]

        print(f"\nИзображение: {os.path.basename(image_path)}")
        print(f"Предсказанная порода: {predicted_breed}")
        print(f"Уверенность: {confidence:.2f}")

        plt.imshow(img)
        plt.title(f"Предсказано: {predicted_breed} ({confidence*100:.1f}%)")
        plt.axis('off')
        plt.show()

        return predicted_breed, confidence

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
        return None, None
    except Exception as e:
        print(f"Произошла ошибка при обработке изображения {image_path}: {e}")
        return None, None

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return False
        improvement = (current_score - self.best_score) if self.mode == 'max' else (self.best_score - current_score)
        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

if __name__ == "__main__":
    print("подготовка данных")
    labels_df = pd.read_csv(LABELS_PATH)
    print(f"Total breeds: {len(labels_df['breed'].unique())}")

    labels_df['filepath'] = labels_df['id'].apply(lambda x: os.path.join(TRAIN_DIR, f'{x}.jpg'))

    label_encoder = LabelEncoder()
    labels_df['encoded_labels'] = label_encoder.fit_transform(labels_df['breed'])
    labels_df['one_hot_labels'] = list(np.eye(NUM_CLASSES, dtype=np.float32)[labels_df['encoded_labels']])

    train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['breed'], random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DogBreedDataGenerator(train_df, IMG_SIZE, NUM_CLASSES, transform=transform)
    val_dataset = DogBreedDataGenerator(val_df, IMG_SIZE, NUM_CLASSES, transform=transform)

    train_generator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_generator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    print("Data generators created.")

    base_model = models.mobilenet_v2(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False
    num_ftrs = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )

    model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=10, mode='max')
    print("\n обучение модели")

    epochs = 20
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_generator:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels).item()
            total_train += inputs.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects / total_train

        # валидация
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_generator:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_running_corrects += torch.sum(preds == labels).item()
                total_val += inputs.size(0)

        val_loss = val_running_loss / total_val
        val_acc = val_running_corrects / total_val

        # scheduler step
        scheduler.step(val_acc)

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Validation accuracy improved to {best_val_acc:.4f}. Model saved to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Ошибка при сохранении модели: {e}")

        if early_stopping(val_acc):
            print("Early stopping triggered.")
            break

    print("\nконец.")
    plot_history(history)

    # загрузка модели
    try:
        loaded_model = models.mobilenet_v2(pretrained=False)
        num_ftrs = loaded_model.classifier[1].in_features
        loaded_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        loaded_model = loaded_model.to(device)
        loaded_model.eval()
        print(f"\nМодель успешно загружена из {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"\nНе удалось загрузить модель: {e}. Возможно, модель еще не была сохранена или путь неверен.")
        print("Используем текущую обученную модель в памяти.")
        loaded_model = model

    print("\nПроверка модели на случайных тестовых изображениях")
    if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
        test_image_files = os.listdir(TEST_DIR)
        random_test_images = np.random.choice(test_image_files, min(5, len(test_image_files)), replace=False)

        for img_file in random_test_images:
            img_path = os.path.join(TEST_DIR, img_file)
            predict_breed(img_path, loaded_model, label_encoder, IMG_SIZE)
    else:
        print(f"Папка с тестовыми изображениями '{TEST_DIR}' не найдена или пуста.")
        print("Пропустите проверку на тестовых изображениях.")
    
    print("\nконец")
