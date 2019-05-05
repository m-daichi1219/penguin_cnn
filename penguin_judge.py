from keras.models import model_from_json
from PIL import Image
import numpy as np

# params
model_file = 'JSONファイルパス'
weight_file = 'hdf5ファイルパス'
image_file = '判定する画像ファイル'
penguin_types = ["アデリーペンギン", "イワトビペンギン", "ガラパゴスペンギン",
                 "キングペンギン", "キンメペンギン", "ケープペンギン",
                 "コウテイペンギン", "コガタペンギン", "ジェンツーペンギン",
                 "シュレーターペンギン", "スネアーズペンギン", "ハネジロペンギン",
                 "ヒゲペンギン", "フィヨルドランドペンギン", "フンボルトペンギン",
                 "マカロニペンギン", "マゼランペンギン", "ロイヤルペンギン"]

# 画像ファイル読み込み
img = Image.open(image_file).convert('RGB')
img = img.resize((300, 300))
img_arr = np.asarray(img)
img_arr = img_arr.reshape(300, 300, 3)
img_arr = img_arr.astype(np.float32) / 255.0    # 正規化
img_arr = np.expand_dims(img_arr, axis=0)

# 予測
model = model_from_json(open(model_file).read())
model.load_weights(weight_file)
ret = model.predict(img_arr)

# 結果
print(penguin_types[np.argmax(ret)])