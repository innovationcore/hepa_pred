import csv
import math
import random
import pandas as pd


class HepADatapoint:
    def __init__(self,
                 age, gender, bmi, hep_c_rna, hep_b, alch, cirr, adm_bili, adm_inr, adm_cr,
                 adm_na, adm_alt, adm_ast, adm_alp, adm_plate, dm, htn, adm_alb, adm_wbc, outcome):
        try:
            if age == '':
                self.age = None
            else:
                self.age = float(age)
            self.gender = 0 if gender == 'F' else 1
            if bmi == '':
                self.bmi = None
            else:
                self.bmi = float(bmi)
            self.hep_c_rna = 0 if hep_c_rna == 'N' else 1
            self.hep_b = 0 if hep_b == 'N' else 1
            if alch == '':
                self.alch = None
            elif alch == 'SOC':
                self.alch = 0
            elif alch == 'MOD':
                self.alch = 1
            elif alch == 'HEAV<6M':
                self.alch = 2
            elif alch == 'HEAV>6M':
                self.alch = 3
            else:
                self.alch = 4
            self.cirr = 0 if cirr == 'N' else 1
            if adm_bili == '':
                self.adm_bili = None
            else:
                self.adm_bili = float(adm_bili)
            if adm_inr == '':
                self.adm_inr = None
            else:
                self.adm_inr = float(adm_inr)
            if adm_cr == '':
                self.adm_cr = None
            else:
                self.adm_cr = float(adm_cr)
            if adm_na == '':
                self.adm_na = None
            else:
                self.adm_na = float(adm_na)
            if adm_alt == '':
                self.adm_alt = None
            else:
                self.adm_alt = float(adm_alt)
            if adm_ast == '':
                self.adm_ast = None
            else:
                self.adm_ast = float(adm_ast)
            if adm_alp == '':
                self.adm_alp = None
            else:
                self.adm_alp = float(adm_alp)
            if adm_plate == '':
                self.adm_plate = None
            else:
                self.adm_plate = float(adm_plate)
            self.dm = 0 if dm == 'N' else 1
            self.htn = 0 if htn == 'N' else 1
            if adm_alb == '':
                self.adm_alb = None
            else:
                self.adm_alb = float(adm_alb)
            if adm_wbc == '':
                self.adm_wbc = None
            else:
                self.adm_wbc = float(adm_wbc)
            self.outcome = 0 if outcome == 'N' else 1
        except Exception as e:
            print(f"{e}")

    def get_x(self):
        return [
            self.age, self.gender, self.bmi, self.hep_c_rna, self.hep_b, self.alch, self.cirr, self.dm, self.htn,
            self.adm_bili, self.adm_inr, self.adm_cr, self.adm_na, self.adm_alt, self.adm_ast, self.adm_alp, self.adm_plate,
            self.adm_alb, self.adm_wbc,
        ]

    def set_x(self, idx, value):
        if idx == 0:
            self.age = value
        if idx == 1:
            self.gender = value
        if idx == 2:
            self.bmi = value
        if idx == 3:
            self.hep_c_rna = value
        if idx == 4:
            self.hep_b = value
        if idx == 5:
            self.alch = value
        if idx == 6:
            self.cirr = value
        if idx == 7:
            self.adm_bili = value
        if idx == 8:
            self.adm_inr = value
        if idx == 9:
            self.adm_cr = value
        if idx == 10:
            self.adm_na = value
        if idx == 11:
            self.adm_alt = value
        if idx == 12:
            self.adm_ast = value
        if idx == 13:
            self.adm_alp = value
        if idx == 14:
            self.adm_plate = value
        if idx == 15:
            self.dm = value
        if idx == 16:
            self.htn = value
        if idx == 17:
            self.adm_alb = value
        if idx == 18:
            self.adm_wbc = value

    def get_y(self):
        return self.outcome

    def has_blanks(self):
        return (True in [x is None or math.isnan(x) for x in self.get_x()])


class HepADataset():
    def __init__(self, csvfile, fill_in_missing: bool = True, initial_partition: float = .8):
        self.data = []
        self.train_data = []
        self.test_data = []
        self.columns = [i for i in range(19)]
        with open(csvfile, 'r') as data_file:
            data_reader = csv.reader(data_file)
            for row in data_reader:
                if row[0] == 'y.AGE':
                    continue
                self.data.append(HepADatapoint(
                    row[0],  row[1],  row[2],  row[3],  row[4],  row[5],  row[6],  row[7],  row[8],  row[9],
                    row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19],
                ))
        if fill_in_missing:
            self.fill_in_nones()
        self.partition_data(initial_partition)

    def set_columns(self, columns: list):
        self.columns = columns

    def partition_data(self, pct: float = .8):
        randomize = random.sample(self.data, len(self.data))
        to_partition = []
        for entry in randomize:
            if not entry.has_blanks():
                to_partition.append(entry)
        self.train_data = to_partition[:int(len(to_partition)*pct)]
        self.test_data = to_partition[int(len(to_partition)*pct):]

    def fill_in_nones(self):
        avgs = [0 for i in range(len(self.data[0].get_x()))]
        counts = [0 for i in range(len(self.data[0].get_x()))]
        for i in range(len(self.data)):
            values = self.data[i].get_x()
            for j in range(len(values)):
                if values[j] is None:
                    continue
                avgs[j] += values[j]
                counts[j] += 1
        for i in range(len(avgs)):
            avgs[i] /= counts[i]
        for i in range(len(self.data)):
            values = self.data[i].get_x()
            for j in range(len(values)):
                if values[j] is None:
                    self.data[i].set_x(j, avgs[j])

    def shuffle_data(self):
        self.train_data = random.sample(self.train_data, len(self.train_data))
        self.test_data = random.sample(self.test_data, len(self.test_data))

    def get_data(self):
        data = []
        for entry in self.data:
            data.append(entry.get_x())
        data_df = pd.DataFrame(data)
        complete = [i for i in range(19)]
        to_remove = list(set(complete) - set(self.columns))
        data_df = data_df.drop(columns=to_remove)
        return data_df.values.tolist()

    def get_training(self):
        x = []
        y = []
        for entry in self.train_data:
            x.append(entry.get_x())
            y.append(entry.get_y())
        x_df = pd.DataFrame(x)
        complete = [i for i in range(19)]
        to_remove = list(set(complete) - set(self.columns))
        x_df = x_df.drop(columns=to_remove)
        return x_df.values.tolist(), y

    def get_testing(self):
        x = []
        y = []
        for entry in self.test_data:
            x.append(entry.get_x())
            y.append(entry.get_y())
        x_df = pd.DataFrame(x)
        complete = [i for i in range(19)]
        to_remove = list(set(complete) - set(self.columns))
        x_df = x_df.drop(columns=to_remove)
        return x_df.values.tolist(), y
