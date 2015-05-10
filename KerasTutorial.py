import numpy as np
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
import math
import time
import operator

weather_st1_coor = (41.995, -87.933)
weather_st2_coor = (41.786, -87.752)

def get_distance(point1, point2):
    (lat1, lon1) = point1
    (lat2, lon2) = point2
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) )
    d = 6373 * c
    return d

def filter_weather_data(line, weather_indexes):
    tmax = float(line[weather_indexes["Tmax"]])
    tmin = float(line[weather_indexes["Tmin"]])
    if str.strip(line[weather_indexes["Tmin"]]) == 'M':
        tavg = (tmax+tmin)/float(2)
    else:
        tavg = float(line[weather_indexes["Tmin"]])
    dew = float(line[weather_indexes["DewPoint"]])
    if str.strip(line[weather_indexes["WetBulb"]]) == 'M':
        wblub = dew + float(8)
    else:
        wblub = float(line[weather_indexes["WetBulb"]])
    if str.strip(line[weather_indexes["PrecipTotal"]]) == 'T':
        prec = 0.005
    elif str.strip(line[weather_indexes["PrecipTotal"]]) == 'M':
        prec = 0
    else:
        prec = float(line[weather_indexes["PrecipTotal"]])
    if str.strip(line[weather_indexes["StnPressure"]]) == 'M':
        pres = 29.3
    else:
        pres = float(line[weather_indexes["StnPressure"]])

    final_line = [tmax, tmin, tavg, dew, wblub, prec, pres]
    return final_line

def get_weather_data():
    weather_dic_st1 = {}
    weather_dic_st2 = {}
    fi = csv.reader(open("input/weather.csv"))
    weather_head = fi.__next__()
    weather_indexes = dict([(weather_head[i], i) for i in range(len(weather_head))])
    for line in fi:
        if line[0] == '1':
            weather_dic_st1[line[1]] = filter_weather_data(line, weather_indexes)
        elif line[0] == '2':
            weather_dic_st2[line[1]] = filter_weather_data(line, weather_indexes)
    final_index = dict([('Tmax',0), ('Tmin',1), ('Tavg',2), ('DewPoint',3), ('WetBulb',4), ('PrecipTotal',5), ('StnPressure',6)])
    return weather_dic_st1, weather_dic_st2, final_index

def get_close_dates(date, no_of_dates, all_dates, backwards):
    if not backwards:
        base_date = time.strptime(date, '%Y-%m-%d')
        all_dates_list = [time.strptime(i, '%Y-%m-%d') for i in all_dates]
        close_dates_tuple_list = [(n, int(d)/86400) for d, n in sorted((abs(time.mktime(x)-time.mktime(base_date)), x) for x in all_dates_list)[:no_of_dates]]
        return dict((time.strftime('%Y-%m-%d', n), d) for (n, d) in close_dates_tuple_list)
    else:
        base_date = time.strptime(date, '%Y-%m-%d')
        all_dates_list = [time.strptime(i, '%Y-%m-%d') for i in all_dates]
        close_dates_tuple_list = [(n, int(d)/86400) for d, n in sorted((abs(time.mktime(x)-time.mktime(base_date)), x) for x in all_dates_list if x < base_date)[:no_of_dates]]
        return dict((time.strftime('%Y-%m-%d', n), d) for (n, d) in close_dates_tuple_list)

def assume_weather_on_date(date, st1_dict, st2_dict):
    print('Assuming weather for {0}'.format(date))
    all_available_dates = set(set(st1_dict.keys())|set(st2_dict.keys()))
    close_dates = get_close_dates(date, 30, all_available_dates, False)
    divide_value = 0
    st1_values = []
    st2_values = []
    for av_date in close_dates:
        distance = close_dates[av_date]
        distance_weight = 100 - distance
        divide_value += distance_weight
        if av_date in st1_dict:
            st1_value_list = st1_dict[av_date]
            multiplied_list = [l*distance_weight for l in st1_value_list]
            if not st1_values:
                st1_values = multiplied_list
            else:
                st1_values = [sum(x) for x in zip(st1_values, multiplied_list)]

        if av_date in st2_dict:
            st2_value_list = st2_dict[av_date]
            multiplied_list = [l*distance_weight for l in st2_value_list]
            if not st2_values:
                st2_values = multiplied_list
            else:
                st2_values = [sum(x) for x in zip(st2_values, multiplied_list)]
    print('Assumed weather : {0} - {1}'.format([value/float(divide_value) for value in st1_values], [value/float(divide_value) for value in st2_values]))
    return [value/float(divide_value) for value in st1_values], [value/float(divide_value) for value in st2_values]

def modify_weather_on_close_days(weather_line, date, number_of_days, st1_data, st2_data):
    all_available_dates = set(set(st1_data.keys())|set(st2_data.keys()))
    close_dates = get_close_dates(date, number_of_days, all_available_dates, True)
    for av_date in close_dates:
        distance = close_dates[av_date]

        if av_date in st1_data:
            mod_value_list = [val/float(distance) for val in st1_data[av_date]]
            weather_line = [sum(x) for x in zip(weather_line, mod_value_list)]

        if av_date in st2_data:
            mod_value_list = [val/float(distance) for val in st2_data[av_date]]
            weather_line = [sum(x) for x in zip(weather_line, mod_value_list)]
    return weather_line

def get_weather_on_loc(st1_dict, st2_dict, location, date):
    dist_to_st1 = get_distance(location, weather_st1_coor)
    dist_to_st2 = get_distance(location, weather_st2_coor)
    if (date in st1_dict) and (date in st2_dict):
        weather_from_st1 = st1_dict[date]
        weather_from_st2 = st2_dict[date]
        weather_line = [((weather_from_st1[i]*dist_to_st1) + (weather_from_st2[i]*dist_to_st2))/float(dist_to_st1+dist_to_st2) for i in range(len(weather_from_st1))]
    if (date not in st1_dict) and (date in st2_dict):
        weather_from_st2 = st2_dict[date]
        weather_line = weather_from_st2
    if (date in st1_dict) and (date not in st2_dict):
        weather_from_st1 = st1_dict[date]
        weather_line = weather_from_st1
    if (date not in st1_dict) and (date not in st2_dict):
        weather_from_st1, weather_from_st2 = assume_weather_on_date(date, st1_dict, st2_dict)
        weather_line = [((weather_from_st1[i]*dist_to_st1) + (weather_from_st2[i]*dist_to_st2))/float(dist_to_st1+dist_to_st2) for i in range(len(weather_from_st1))]

    return weather_line

def process_line(line, indexes, st1_dic, st2_dic, weather_indexes):
    def get(name):
        return line[indexes[name]]

    date = get("Date")
    month = float(date.split('-')[1])
    week = int(date.split('-')[1]) * 4 + int(date.split('-')[2]) / 7
    latitude = float(get("Latitude"))
    longitude = float(get("Longitude"))
    location = (latitude, longitude)
    weather_on_loc = get_weather_on_loc(st1_dic, st2_dic, location, date)
    mod_weather_on_loc = modify_weather_on_close_days(weather_on_loc, date, 100, st1_dic, st2_dic)
    tmax = float(mod_weather_on_loc[weather_indexes["Tmax"]])
    tmin = float(mod_weather_on_loc[weather_indexes["Tmin"]])
    tavg = float(mod_weather_on_loc[weather_indexes["Tavg"]])
    dewpoint = float(mod_weather_on_loc[weather_indexes["DewPoint"]])
    wetbulb = float(mod_weather_on_loc[weather_indexes["WetBulb"]])
    precip = float(mod_weather_on_loc[weather_indexes["PrecipTotal"]])
    pressure = float(mod_weather_on_loc[weather_indexes["StnPressure"]])

    return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, precip, pressure]

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(input_dim, 32, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, 32, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, 32, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #
    # model.add(Dense(64, 128, init='lecun_uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))
    # #
    # model.add(Dense(128, 256, init='lecun_uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(32, output_dim, init='lecun_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta")
    return model


# now the actual script

print("Processing training data...")

rows = []
labels = []
temp_rows = []
fi = csv.reader(open("input/train.csv"))
head = fi.__next__()
indexes = dict([(head[i], i) for i in range(len(head))])
weather_st1_dic, weather_st2_dic, weather_indexes = get_weather_data()
final_headers = ['month', 'week', 'lat', 'lon', 'tmax', 'tmin', 'tavg', 'dew', 'wbulb', 'prec', 'pres', 'wnvPresent']
lineNo=1
for line in fi:
    print(lineNo)
    rows.append(process_line(line, indexes, weather_st1_dic, weather_st2_dic, weather_indexes))
    labels.append(float(line[indexes["WnvPresent"]]))
    processed_data = process_line(line, indexes, weather_st1_dic, weather_st2_dic, weather_indexes)
    processed_data.append(float(line[indexes["WnvPresent"]]))
    temp_rows.append(processed_data)
    lineNo+=1

#fot = csv.writer(open('output/reg.csv', 'w'), lineterminator="\n")
#fot.writerow(final_headers)
#for l in temp_rows:
    #fot.writerow(l)

X = np.array(rows)
y = np.array(labels)

X, y = shuffle(X, y)
X, scaler = preprocess_data(X)
Y = np_utils.to_categorical(y)

input_dim = X.shape[1]
output_dim = 2

print("Validation...")

nb_folds = 10
kfolds = KFold(len(y), nb_folds)
av_roc = 0.
f = 0
for train, valid in kfolds:
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    X_train = X[train]
    X_valid = X[valid]
    Y_train = Y[train]
    Y_valid = Y[valid]
    y_valid = y[valid]

    print("Building model...")
    model = build_model(input_dim, output_dim)

    print("Training model...")

    model.fit(X_train, Y_train, nb_epoch=100, batch_size=16, validation_data=(X_valid, Y_valid), verbose=0)
    valid_preds = model.predict_proba(X_valid, verbose=0)
    valid_preds = valid_preds[:, 1]
    roc = metrics.roc_auc_score(y_valid, valid_preds)
    print("ROC:", roc)
    av_roc += roc

print('Average ROC:', av_roc/nb_folds)

print("Generating submission...")

#model = build_model(input_dim, output_dim)
#model.fit(X, Y, nb_epoch=100, batch_size=16, verbose=0)

fi = csv.reader(open("input/test.csv"))
head = fi.__next__()
indexes = dict([(head[i], i) for i in range(len(head))])
rows = []
ids = []
for line in fi:
    rows.append(process_line(line, indexes, weather_st1_dic, weather_st2_dic, weather_indexes))
    ids.append(line[0])
X_test = np.array(rows)
X_test, _ = preprocess_data(X_test, scaler)

preds = model.predict_proba(X_test, verbose=0)

fo = csv.writer(open("output/keras-nn.csv", "w"), lineterminator="\n")
fo.writerow(["Id","WnvPresent"])

for i, item in enumerate(ids):
    fo.writerow([ids[i], preds[i][1]])
