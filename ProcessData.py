__author__ = 'Sulantha'
import csv
import math
import time


weather_st1_coor = (41.995, -87.933)
weather_st2_coor = (41.786, -87.752)

species_list=[]

def get_distance(point1, point2):
    (lat1, long1) = point1
    (lat2, long2) = point2
    degrees_to_radians = math.pi / 180.0

    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)

    return arc * 6373


def filter_weather_data(line, weather_indexes):
    tmax = float(line[weather_indexes["Tmax"]])
    tmin = float(line[weather_indexes["Tmin"]])
    if str.strip(line[weather_indexes["Tmin"]]) == 'M':
        tavg = (tmax + tmin) / float(2)
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
    final_index = dict([('Tmax', 0), ('Tmin', 1), ('Tavg', 2), ('DewPoint', 3), ('WetBulb', 4), ('PrecipTotal', 5),
                        ('StnPressure', 6)])
    return weather_dic_st1, weather_dic_st2, final_index


def get_close_dates(date, no_of_dates, all_dates, backwards):
    if not backwards:
        base_date = time.strptime(date, '%Y-%m-%d')
        all_dates_list = [time.strptime(i, '%Y-%m-%d') for i in all_dates]
        close_dates_tuple_list = [(n, int(d) / 86400) for d, n in
                                  sorted((abs(time.mktime(x) - time.mktime(base_date)), x) for x in all_dates_list)[
                                  :no_of_dates]]
        return dict((time.strftime('%Y-%m-%d', n), d) for (n, d) in close_dates_tuple_list)
    else:
        base_date = time.strptime(date, '%Y-%m-%d')
        all_dates_list = [time.strptime(i, '%Y-%m-%d') for i in all_dates]
        close_dates_tuple_list = [(n, int(d) / 86400) for d, n in sorted(
            (abs(time.mktime(x) - time.mktime(base_date)), x) for x in all_dates_list if x < base_date)[:no_of_dates]]
        return dict((time.strftime('%Y-%m-%d', n), d) for (n, d) in close_dates_tuple_list)


def assume_weather_on_date(date, st1_dict, st2_dict):
    print('Assuming weather for {0}'.format(date))
    all_available_dates = set(set(st1_dict.keys()) | set(st2_dict.keys()))
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
            multiplied_list = [l * distance_weight for l in st1_value_list]
            if not st1_values:
                st1_values = multiplied_list
            else:
                st1_values = [sum(x) for x in zip(st1_values, multiplied_list)]

        if av_date in st2_dict:
            st2_value_list = st2_dict[av_date]
            multiplied_list = [l * distance_weight for l in st2_value_list]
            if not st2_values:
                st2_values = multiplied_list
            else:
                st2_values = [sum(x) for x in zip(st2_values, multiplied_list)]
    print('Assumed weather : {0} - {1}'.format([value / float(divide_value) for value in st1_values],
                                               [value / float(divide_value) for value in st2_values]))
    return [value / float(divide_value) for value in st1_values], [value / float(divide_value) for value in st2_values]


def modify_weather_on_close_days(weather_line, date, number_of_days, st1_data, st2_data):
    all_available_dates = set(set(st1_data.keys()) | set(st2_data.keys()))
    close_dates = get_close_dates(date, number_of_days, all_available_dates, True)
    for av_date in close_dates:
        distance = close_dates[av_date]

        if av_date in st1_data:
            mod_value_list = [val / float(distance) for val in st1_data[av_date]]
            weather_line = [sum(x) for x in zip(weather_line, mod_value_list)]

        if av_date in st2_data:
            mod_value_list = [val / float(distance) for val in st2_data[av_date]]
            weather_line = [sum(x) for x in zip(weather_line, mod_value_list)]
    return weather_line


def get_weather_on_loc(st1_dict, st2_dict, location, date):
    dist_to_st1 = get_distance(location, weather_st1_coor)
    dist_to_st2 = get_distance(location, weather_st2_coor)
    if (date in st1_dict) and (date in st2_dict):
        weather_from_st1 = st1_dict[date]
        weather_from_st2 = st2_dict[date]
        weather_line = [((weather_from_st1[i] * dist_to_st1) + (weather_from_st2[i] * dist_to_st2)) / float(
            dist_to_st1 + dist_to_st2) for i in range(len(weather_from_st1))]
    if (date not in st1_dict) and (date in st2_dict):
        weather_from_st2 = st2_dict[date]
        weather_line = weather_from_st2
    if (date in st1_dict) and (date not in st2_dict):
        weather_from_st1 = st1_dict[date]
        weather_line = weather_from_st1
    if (date not in st1_dict) and (date not in st2_dict):
        weather_from_st1, weather_from_st2 = assume_weather_on_date(date, st1_dict, st2_dict)
        weather_line = [((weather_from_st1[i] * dist_to_st1) + (weather_from_st2[i] * dist_to_st2)) / float(
            dist_to_st1 + dist_to_st2) for i in range(len(weather_from_st1))]

    return weather_line

def get_species_id(species_type):
    if species_type not in species_list:
        species_list.append(species_type)
    return species_list.index(species_type)

def process_line(line, st1_dic, st2_dic, weather_indexes, type, wnvPresentDict={}):

    if type == 'test':
        date = line[1]
    else:
        date = line[0]
    year = float(date.split('-')[0])
    month = float(date.split('-')[1])
    date_number = float(date.split('-')[2])
    week = int(date.split('-')[1]) * 4 + int(date.split('-')[2]) / 7
    if type == 'test':
        species = float(get_species_id(line[3]))
        latitude = float(line[8])
        longitude = float(line[9])
        addAcc = float(line[10])
    else:
        species = float(get_species_id(line[2]))
        latitude = float(line[7])
        longitude = float(line[8])
        addAcc = float(line[9])

    location = (latitude, longitude)
    weather_on_loc = get_weather_on_loc(st1_dic, st2_dic, location, date)
    mod_weather_on_loc = modify_weather_on_close_days(weather_on_loc, date, 25, st1_dic, st2_dic)
    tmax = float(mod_weather_on_loc[weather_indexes["Tmax"]])
    tmin = float(mod_weather_on_loc[weather_indexes["Tmin"]])
    tavg = float(mod_weather_on_loc[weather_indexes["Tavg"]])
    dewpoint = float(mod_weather_on_loc[weather_indexes["DewPoint"]])
    wetbulb = float(mod_weather_on_loc[weather_indexes["WetBulb"]])
    precip = float(mod_weather_on_loc[weather_indexes["PrecipTotal"]])
    pressure = float(mod_weather_on_loc[weather_indexes["StnPressure"]])


    if type == 'train':
        wnvPresent = float(wnvPresentDict[(date, str(get_species_id(line[2])), line[7], line[8])])
        wnvPresent = float(line[11])
        return [year, month, week, date_number, species, latitude, longitude, addAcc, tmax, tmin, tavg, dewpoint, wetbulb, precip, pressure, wnvPresent]
    if type == 'test':
        return [year, month, week, date_number, species, latitude, longitude, addAcc, tmax, tmin, tavg, dewpoint, wetbulb, precip, pressure]


weather_st1_dic, weather_st2_dic, weather_indexes = get_weather_data()
fi = csv.reader(open("input/train.csv").read().splitlines())
head = fi.__next__()

wnvPresentDict={}

for line in fi:
    spec_id = get_species_id(line[2])
    if (line[0], str(spec_id), line[7], line[8]) in wnvPresentDict:
        wnvValue = wnvPresentDict[(line[0], str(spec_id), line[7], line[8])]
        if (float(wnvValue) is 1.0) or (float(line[11]) is 1.0):
            wnvPresentDict[(line[0], str(spec_id), line[7], line[8])] = str(1.0)
    else:
        wnvPresentDict[(line[0], str(spec_id), line[7], line[8])] = line[11]

print('Size of WNV PRESENT DICT: {0}'.format(len(wnvPresentDict)))

fi = csv.reader(open("input/train.csv").read().splitlines())
head = fi.__next__()
trainingData = []
for record in fi:
    trainingData.append(process_line(record, weather_st1_dic, weather_st2_dic, weather_indexes, 'train', wnvPresentDict))
print('Trianing Data Size: {0}'.format(len(trainingData)))

final_headers = ['Year', 'Month', 'Week', 'Date', 'Species', 'lat', 'lon', 'addAcc', 'tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'prec', 'pressure', 'wnvPresent']
fot_f = open('output/processed_training.csv', 'w')
fot = csv.writer(fot_f, lineterminator="\n")
fot.writerow(final_headers)
for l in trainingData:
    fot.writerow(l)
fot_f.close()
print('Training file writing complete')
ft = csv.reader(open("input/test.csv").read().splitlines())
head = ft.__next__()
testData = []
for record in ft:
    testData.append(process_line(record, weather_st1_dic, weather_st2_dic, weather_indexes, 'test'))
final_headers_test = ['Year', 'Month', 'Week', 'Date', 'Species', 'lat', 'lon', 'addAcc', 'tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'prec', 'pressure']
fott = csv.writer(open('output/processed_test.csv', 'w'), lineterminator="\n")
fott.writerow(final_headers_test)
for l in testData:
    fott.writerow(l)
print('Test file writing complete')