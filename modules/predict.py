import dill
import pandas as pd
import os
import json
import glob
import datetime

path = os.environ.get('PROJECT_PATH', '.')


def load_model():

    files_path = os.path.join(f'{path}/data/models/', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)

    with open(files[0], 'rb') as f:
        object_to_load = dill.load(f)
        return object_to_load


def predict():
    predict_df = dict()
    model = load_model()

    for filename in os.listdir(f"{path}/data/test"):
        name = os.path.basename(filename).split('.')[0]
        with open(os.path.join(f"{path}/data/test", filename), 'r') as f:
            text = json.load(f)
            df = pd.DataFrame(text, index=[0])
            y = model.predict(df)
            predict_df[name] = predict_df.get(name, y[0])

    final_predict = pd.DataFrame(predict_df.items(), columns=['id', 'predict'])
    final_predict.to_csv(f'{path}/data/predictions/cars_pred_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
