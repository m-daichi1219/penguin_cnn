import glob
import random
import math
from PIL import Image
import numpy as np

# 全データ格納用配列
allfiles = []
# 種類の合計
type_total = None


def make_image(files):
    # 画像データ用配列
    X = []
    # ラベルデータ用配列
    Y = []
    cnt = 0
    T = np.zeros((len(files), type_total))
    for label, fdata in files:
        # 画像ファイル正規化
        img = Image.open(fdata).convert('RGB')
        img = img.resize((300, 300))
        data = np.asarray(img)
        data = data.astype(np.float32) / 255.0  # 画像のピクセル値を0.0~1.0に正規化する
        X.append(data)

        # one-hotラベル作成
        T[cnt][label] = 1
        Y.append(T[cnt])
        cnt += 1
        # Y.append(label)

    return np.array(X), np.array(Y , dtype = 'uint8')


def make_one_hot(labels):
    T = np.zeros((labels.size, 18))
    for idx, row in enumerate(T):
        row[labels[idx]] = 1

    return T


def load_data(root_dir, types):
    global allfiles, type_total
    type_total = len(types)

    for idx, type in enumerate(types):
        image_dir = root_dir + "\\" + type
        files = glob.glob(image_dir + "/*.jpeg")

        for file in files:
            allfiles.append((idx, file))

    # 訓練データとテストデータに分ける
    random.shuffle(allfiles)
    th = math.floor(len(allfiles) * 0.8)
    learning_data = allfiles[0:th]
    test_data = allfiles[th:]
    # 画像の正規化,one-hotラベルの作成
    x_train, t_train = make_image(learning_data)
    x_test, t_test = make_image(test_data)

    # 訓練画像, 訓練ラベル, テスト画像, テストラベル
    return (x_train, t_train), (x_test, t_test)
