# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:54:15 2020

@author: Leo Turowski, Luciano Melodia
@executive author: Noah Becker
"""

import csv
import os
import ntpath
import glob
import pandas as pd
import numpy as np
import pickle
import typing
import math
import matplotlib
import matplotlib.pyplot as plt
import timeSeriesConfig as cfg
import tifffile
import statistics

from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField as GAF
from pyts.datasets import load_gunpoint
from numpy.lib import recfunctions as rfn
from itertools import groupby
from zipfile import ZipFile
from sklearn.impute import SimpleImputer
from scipy.stats import entropy
from PIL import Image


def remove_nans(directory: str, fileextension: str = ".tiff", size: int = 500):
    """
    **Remove nan values in custom dataset.**

    This function just removes the nan values from our custom `.tiff`-files.

    + param **directory**: directory of the files, dtype `str`.
    + param **fileextension**: extension, dtype `str`.
    + param **size**: width of the images, dtype `int`.
    """
    result = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(directory)
        for f in filenames
        if os.path.splitext(f)[1] == fileextension
    ]

    for i in result:
        img0 = np.array(Image.open(i))[0]
        img1 = np.array(Image.open(i))[1]
        img2 = np.array(Image.open(i))[2]
        imp_mean0 = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=np.nanmedian(img0)
        )
        imp_mean1 = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=np.nanmedian(img1)
        )
        imp_mean2 = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=np.nanmedian(img2)
        )
        value0 = imp_mean0.fit_transform(img0.transpose().reshape(1, -1)).reshape(size)
        value1 = imp_mean1.fit_transform(img1.transpose().reshape(1, -1)).reshape(size)
        value2 = imp_mean2.fit_transform(img2.transpose().reshape(1, -1)).reshape(size)
        image = np.array([value0, value1, value2])


def zip_to_csv(path: str):
    """
    **Converts a packed `sql` file to a `csv` file.**

    This function unpacks the specified `zip` file at its location and calls the specified `sql_to_csv`
    function on the `sql` file, with the same name as the zip file.

    + param **path**: as an absolute path to the `zip` file with the `sql` in it, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "zip":
        print("this is not a zip file")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    with ZipFile(filename, "r") as zip:
        zip.extractall()
    sql_to_csv(path[:-4])


def zip_to_npy(path: str):
    """
    **Converts a packed `sql` file to an npy file.**

    This function unpacks the specified zip file at its location and calls the specified sql_to_npy
    function on the `sql` file, with the same name as the zip file.

    + param **path**: as an absolute path to the zip file containing `sql`, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "zip":
        print("this is not a zip file")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    with ZipFile(filename, "r") as zip:
        zip.extractall()
    sql_to_npy(path[:-4])


def sql_to_csv(path: str, delimiter: str = "\n"):
    """
    **Converts a set of `INSERT` statements to `csv` format.**

    Extracts the data from a set of `INSERT` statements stored in a `sql` file, this function
    converts the data into a `csv` file, where each non `INSERT` line is stored in a separate pickle file
    file, and the data of the `INSERT` statements is stored line by line, with the specified delimiter
    at the end of each line.

    + param **path**: as absolute path to the `sql` file, type `str`.
    + param **delimiter**: as delimiter at the end of each line, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("This path does not lead to a file.")
        return
    if path[-3:] != "sql":
        print("This is not an `sql` file.")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    with open(filename, "r") as oldfile:
        with open(filename[:-3] + "csv", "w", newline="") as newfile:
            content = oldfile.readlines()
            data = []
            picklelist = []
            for line in content:
                if line.startswith("I"):
                    line = line.split("(")
                    line = line[1]  # Cuts of the `INSERT` part of the `sql` statement.
                    line = line.split(
                        ")"
                    )  # Cuts of the ");\n" end of the `sql` statement.
                    line = line[0]
                    line = line.replace("'", "")
                    data.append(line)
                else:
                    picklelist.append(line)

            write = csv.writer(newfile, delimiter=delimiter)
            write.writerow(data)
            pickle.dump(picklelist, open((filename[:-3] + "p"), "wb"))


def sql_to_npy(path: str, delimiter: str = ","):
    """
    **Converts a set of `INSERT` statements into a numpy array.**

    Similar to the csv function, this function also stores unused data in a pickle file and creates
    a brand new file with the extracted data, this time in npy format, but this time the
    the delimiter must be the delimiter used in the `sql` file, plus an additional
    missing_values string used to represent missing data.

    + param **path**: as the absolute path to the `sql` file, type `str`.
    + param **delimiter**: as the string used in the `sql` file to separate the data, type `str`.
    + param **missing_values**: the string used for missing data, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("This path does not lead to a file.")
        return
    if path[-3:] != "sql":
        print("This is not an `sql` file.")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    with open(filename, "r") as oldfile:
        newfilename = filename[:-3]
        content = oldfile.readlines()
        data = []
        picklelist = []
        for line in content:
            if line.startswith("I"):
                line = line.split("(")
                line = line[1]  # Cuts of the `INSERT` part of the `sql` statement
                line = line.split(")")  # Cuts of the ");\n" end of the `sql` statement.
                line = line[0]
                line = line.replace("'", "")
                data.append(line)
            else:
                picklelist.append(line)
        nparray = np.loadtxt(
            data, dtype=str, delimiter=delimiter, encoding="ASCII", ndmin=2
        )
        np.save(newfilename + "npy", nparray)
        pickle.dump(picklelist, open(newfilename + "p", "wb"))


def csv_to_sql(path: str, delimiter: str = "\n"):
    """
    **Converts an `csv` file into a set of `INSERT` statements.**

    This function converts each set of data separated by the specified separator character
    of a `csv` file into an `INSERT` statement. It also inserts data
    stored in a pickle file which has the same name as the `csv` file,
    as a comment at the beginning, so as not to interfere with functionality.

    + param **path**: as absolute path to the `csv` file, type `str`.
    + param **delimiter**: as string to recognize the different records, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("This path does not lead to a file")
        return
    if path[-3:] != "csv":
        print("this is not a `csv` file")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    with open(path, newline="") as oldfile:
        newfilename = filename[:-3]
        picklelist = pickle.load(open(newfilename + "p", "rb"))
        table = picklelist[0]
        table = table[table.rfind(" ") + 1 : -1]
        reader = csv.reader(oldfile, delimiter=delimiter)
        with open(newfilename + "sql", "w") as newfile:
            newfile.writelines(picklelist)
            for line in reader:
                line = "".join(line)
                line = line.replace(",", "','")
                newfile.write("`INSERT` INTO %s VALUES('" % table)
                newfile.write("%s');\n" % line)


def csv_to_npy(path: str, delimiter: str = ","):
    """
    **Converts a `csv` file to a Numpy array representation.**

    This function converts a `csv` file into a 2-dimensional Numpy representation,
    where each record separated by the specified delimiter is interpreted as a new line.

    + param **path**: as absolute path to the `csv` file, type `str`.
    + param **delimiter**: the string used to determine the rows of the numpy array, type `str`.
    + param **missing_values**: as the string used to represent missing data, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "csv":
        print("this is not a `csv` file")
        return
    os.chdir(os.path.dirname(path))
    filename = ntpath.basename(path)
    newfilename = filename[:-3] + "npy"
    data = np.loadtxt(path, dtype=str, delimiter=delimiter, encoding="ASCII", ndmin=2)
    np.save(newfilename, data)


def npy_to_sql(path: str):
    """
    **Convert a npy file into a set of `INSERT` statements.**

    This function is the reverse function to sql_to_npy and when used in conjuction
    you end up with the same file in the end as you had in the beginning.

    + param **path**: as the absolute path to the npy file, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "npy":
        print("this is not an npy file")
        return
    os.chdir(os.path.dirname(path))
    np_array = np.load(path, "r")
    filename = ntpath.basename(path)
    with open(filename[:-3] + "sql", "w") as newfile:
        picklelist = pickle.load(open(filename[:-3] + "p", "rb"))
        newfile.writelines(picklelist)
        table = picklelist[0]
        table = table[table.rfind(" ") + 1 : -1]
        for row in np_array:
            data = ",".join(row)
            data += "'"
            data = data.replace(",", "','")
            data = data.replace("'NULL'", "NULL")
            newfile.write("`INSERT` INTO {0} VALUES('{1});\n".format(table, data))


def npy_to_csv(path: str):
    """
    **Converts an npy file into a series of `INSERT` statements.**

    This function is the reverse of sql_to_npy and if you use it in conjunction with
    you will have the same file at the end as at the beginning.

    + param **path**: as absolute path to the npy file, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "npy":
        print("this is not an npy file")
        return
    os.chdir(os.path.dirname(path))
    np_array = np.load(path, "r")
    filename = ntpath.basename(path)
    with open(filename[:-3] + "csv", "w") as newfile:
        for row in np_array:
            data = ",".join(row)
            newfile.write("{0}\n".format(data))


def gen_GAF(path: str):
    """
    **Generate a gramian angle field with user input.**

    This function receives user input via the console to generate
    either a Gramian Angular Summation Field or a Gramian Angular Difference Field
    from the data of a Numpy array using the function gen_GAF_exec.

    + param **path**: as absolute path to the npy file, type `str`.
    """
    path = checkpath(path)
    if not (os.path.isfile(path)):
        print("this path does not lead to a file")
        return
    if path[-3:] != "npy":
        print("this is not an npy file")
        return
    os.chdir(os.path.dirname(path))
    np_array = np.load(path, encoding="ASCII")
    method = int(
        input(
            "Enter if you either want a Summation field(1) or a Difference field(2):\n"
        )
    )
    if method == 1:
        method = "summation"
    else:
        method = "difference"
    null_value = input(
        "Enter the number you want to represent missing/NULL values (Default: 0):\n"
    )
    gen_GAF_exec(np_array, (-1, 1), method, null_value)


def gen_GAF_exec(
    data: list,
    sample_range: None or tuple = (-1, 1),
    method: str = "summation",
    null_value: str = "0",
):
    """
    **Generate a Gramian Angular Field.**

    This is the actual function when it comes to generating a Gramian Angular Field
    from the data of a Numpy array. This function takes several variables to determine
    how the field should be scaled, what size it should have
    and whether it is a sum or difference field.

    + param **data**: as the contents of an npy file , type `list`.
     param **size**: this is the size of the square output image, type int or `float`.
    + param **sample_range**: as the range to scale the data to, type None or `tuple`.
    + param **method**: as the type of field to scale to, type `sum` or `difference`, type `str`.
    + param **null_value**: as the number to use instead of NULL, type `str`.
    """
    gaf = GAF(sample_range=sample_range, method=method)
    data = np.where(data == "NULL", null_value, data)
    data = data[:, 3:].astype(dtype=float)
    data_gaf = gaf.fit_transform(data)
    plt.imshow(data_gaf[0], cmap="rainbow", origin="lower")
    plt.show()


def false_input():
    """
    **Output error and return to main.**

    This function prints an error message to the console and invokes main.
    """
    print("This is an invalid option.")
    main()


def exit():
    """
    **Print message and exit program.**

    This function prints a message on the console and terminates the program.
    """
    print("Thank you for using shapy, the converter of your choice.")
    exit(0)


def switchoption(n: int, path: str):
    """
    **Call a function.**

    This function calls one of the functions of this program according to the `n`.
    and gives it the path as input.

    + param **n**: this number specifies which function to call, type `int`.
    + param **path**: this is the path to the file used for the function to be called, type `str`.
    """
    switcher = {
        1: zip_to_csv,
        2: zip_to_npy,
        3: sql_to_csv,
        4: sql_to_npy,
        5: csv_to_sql,
        6: csv_to_npy,
        7: npy_to_sql,
        8: npy_to_csv,
        9: gen_GAF,
        0: exit,
    }
    function = switcher.get(n, false_input)
    function(path)


def checkpath(path: str):
    """
    **Check the path if it is relative.**

    This function removes all quotes from a path and checks whether it is relative or absolute
    it returns a *cleaned* path which is the absolute representation of the given path.

    + param **path**: the string to use as path, type `str`.
    + return **path**: the absolute path, type `str`.
    """
    path = path.replace('"', "")
    path = path.replace("'", "")
    if os.path.isabs(path):
        return path
    return os.getcwd + path


def split_csv_file(
    path: str, header: bool = False, column: int = 1, delimiter: str = ";"
) -> bool:
    """
    **Splits a `csv` file according to a column.**

    Takes as input a `csv` file path. Groups the file according to a certain label and stores the
    data into multiple files named after the label.

    + param **path**: path to the desired `csv` file, type `str`.
    + param **header**: whether to remove the header or not, type `bool`.
    + param **column**: index of the desired column to group by, type `bool`.
    + param **delimiter**: delimiter for the `csv` file, default `;`, type `str`.
    """
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)

        if not header:
            next(reader)

        lst = sorted(reader, key=lambda x: x[column])
        groups = groupby(lst, key=lambda x: x[column])
        basename = os.path.splitext(os.path.basename(path))[0]
        destiny = cfg.paths["test"] + "/" + basename

        try:
            os.mkdir(destiny)
        except OSError:
            print("Creation of the directory %s failed" % destiny)
        else:
            print("Successfully created the directory %s " % destiny)

        for k, g in groups:
            filename = destiny + "/" + basename + k + ".csv"
            with open(filename, "w", newline="") as fout:
                csv_output = csv.writer(fout)
                for line in g:
                    csv_output.writerow(line)


def create_datasets(root: str):
    """
    **Splits the various files into directories.**

    This function recursively lists all files in a directory and divides them into the
    hard-coded folder structure for persistence diagrams, heat kernels, the embeddings, the
    persistent silhouette, and the Betti curves.

    + param: root, type `str`.
    """
    list_of_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            file = os.path.join(path, name)

            if "_persistence_diagram.npy" in file:
                os.replace(file, cfg.paths["general"] + "/powerplant_pershom/" + name)
            elif "_heat_kernel.npy" in file:
                os.replace(
                    file, cfg.paths["general"] + "/powerplant_heatkernels/" + name
                )
            elif "_betti_curve.npy" in file:
                os.replace(
                    file, cfg.paths["general"] + "/powerplant_betticurve/" + name
                )
            elif "_embedded_" in file:
                os.replace(
                    file, cfg.paths["general"] + "/powerplant_slidingwindow/" + name
                )
            elif "_silhouette.npy" in file:
                os.replace(
                    file, cfg.paths["general"] + "/powerplant_silhouette/" + name
                )
            else:
                pass
    print("Done. Subdivided all files into respective directories.")
