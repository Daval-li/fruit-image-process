# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:27:43 2020

@author: 1
"""
import csv

key_data = ["image file", "fruit name"]

def save_fruitinfo(file_path, data):
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    csv_file = open(file_path, "w", newline="")
    value_data = [target for target in data]
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(key_data)
    csv_writer.writerows(value_data)
    csv_file.close()






