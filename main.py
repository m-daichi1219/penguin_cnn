import sys, os
sys.path.append(os.pardir)
from create_data import *
from keras import layers, models
from keras.layers import Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2)) # ドロップアウト率0.2を追加

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2)) # ドロップアウト率0.2を追加

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2)) # ドロップアウト率0.2を追加

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2)) # ドロップアウト率0.2を追加

model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(Dropout(0.5)) # ドロップアウト率0.5を追加

model.add(layers.Dense(18,activation="softmax"))
adam = Adam(lr=1e-4)

model.summary()


model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

root_dir = "画像フォルダのルートとなるフォルダ"

penguin_types = ["アデリーペンギン", "イワトビペンギン", "ガラパゴスペンギン",
                 "キングペンギン", "キンメペンギン", "ケープペンギン",
                 "コウテイペンギン", "コガタペンギン", "ジェンツーペンギン",
                 "シュレーターペンギン", "スネアーズペンギン", "ハネジロペンギン",
                 "ヒゲペンギン", "フィヨルドランドペンギン", "フンボルトペンギン",
                 "マカロニペンギン", "マゼランペンギン", "ロイヤルペンギン"]

# データの読み込み
# 訓練画像, 訓練ラベル, テスト画像, テストラベル
(x_train, t_train), (x_test, t_test) = load_data(root_dir, penguin_types)

model = model.fit(x_train, t_train, epochs=10, batch_size=32, validation_data=(x_test,t_test))

# 学習結果を表示
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('acc')
plt.legend()
plt.savefig('penguin_acc')

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('loss')
plt.legend()
plt.savefig('penguin_loss')

# モデルの保存
json_string = model.model.to_json()
open('PenguinsProblemProject.json', 'w').write(json_string)

# 重みの保存
hdf5_file = "PenguinsProblemProject.hdf5"
model.model.save_weights(hdf5_file)

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))