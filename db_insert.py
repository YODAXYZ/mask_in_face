# import psycopg2
# import logging
# import datetime
# import numpy as np
#
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
# from PIL import Image
# import pickle
#
#
# def img_to_array():
#     img = Image.open('images/test.JPG')
#     numpydata = np.asarray(img)
#     return numpydata
#
#
# def np_to_byte(np_array):
#     pickle.dumps(np_array)
#
#
# class DbWorker:
#     def __init__(self):
#         try:
#             self.__conn = psycopg2.connect(dbname='itis_java', user='postgres',
#                                            password='postgres', host='localhost')
#             self.__conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
#             self.__cursor = self.__conn.cursor()
#             logging.getLogger().addHandler(logging.StreamHandler())
#         except:
#             logging.exception("DB connection failed")
#
#     def insert_data(self, time, image):
#         try:
#             self.__cursor.execute(f"""
#                INSERT INTO violator(time_shot, photo) VALUES ('{time}', {pickle.dumps(image)})
#             """)
#         except:
#             logging.exception("Insert into continent table error !")
#
#
# dbWorker = DbWorker()
# A = [[1, 4]]
# dbWorker.insert_data(datetime.datetime.now(), A)



import mysql.connector
from mysql.connector import Error


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def insertBLOB(emp_id, name, photo, biodataFile):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='python_db',
                                             user='root',
                                             password='pynative@#29')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO python_employee
                          (id, name, photo, biodata) VALUES (%s,%s,%s,%s)"""

        empPicture = convertToBinaryData(photo)
        file = convertToBinaryData(biodataFile)

        # Convert data into tuple format
        insert_blob_tuple = (emp_id, name, empPicture, file)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into python_employee table", result)

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
