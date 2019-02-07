# sql1.py
"""Volume 3: SQL 1 (Introduction).
<Name> Natalie Larsen
<Class> 001
<Date> 11-14-2018
"""

import sqlite3 as sql
import csv
import numpy as np
from matplotlib import pyplot as plt


# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    conn = sql.connect(db_file)
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Drop MajorInfo, CourseInfo, StudentInfo and StudentGrades if they exist
            cur.execute("DROP TABLE IF EXISTS MajorInfo")
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")

            #Create MajorInfo, CourseInfo, StudentInfo and StudentGrades tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

            #Insert info for MajorInfo and CourseInfo
            major_rows = [(1,"Math"),(2,"Science"),(3,"Writing"),(4,"Art")]
            course_rows = [(1,"Calculus"),(2,"English"),(3,"Pottery"),(4,"History")]
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", major_rows)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", course_rows)
            #Retrieve and insert info for StudentInfo and StudentGrades
            with open("student_info.csv") as infile:
                info_rows = list(csv.reader(infile))
            with open("student_grades.csv") as infile:
                grades_rows = list(csv.reader(infile))
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", info_rows)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", grades_rows)

            #Change MajorID from -1 to NULL
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1")


    finally:
        conn.close()



# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            #Clear out USEarthquakes
            cur.execute("DROP TABLE IF EXISTS USEarthquakes")
            #Create new USEarthquakes table with Year, Month, Day, Hour, Minute, Second, Latitude, Longitude and Magnitude columns
            cur.execute("CREATE TABLE USEarthquakes"
                        "(Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, "
                        "Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
            #Read and insert info from file
            with open(data_file) as infile:
                rows = list(csv.reader(infile))
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)
            #Delete magnitude 0 entries and set 0 to NULL in Day, Hour, Minute and Second column
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0")
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0")

    finally:
        conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    good_grades = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Find all names and classes where the grade is A or A+
            good_grades = list(cur.execute("SELECT SI.StudentName, CI.CourseName "
                                            "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                                            "WHERE SI.StudentID == SG.StudentID AND CI.CourseID == SG.CourseID AND SG.Grade IN ('A','A+')"))
            cur.fetchall()

    finally:
        conn.close()

    #return results of search
    return good_grades

# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    mag_19 = []
    mag_20 = []
    average = 0
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #find list of magitudes from years 1800-1899 abd 1900-1999
            mag_19 = list(cur.execute("SELECT Magnitude FROM USEarthquakes "
                                      "WHERE Year IN (1800,1899);"))
            mag_20 = list(cur.execute("SELECT Magnitude FROM USEarthquakes "
                                      "WHERE Year IN (1900,1999);"))
            #find average of all the magnitudes
            average = list(mag_20+mag_19)
            average = sum(np.ravel(average))/len(average)
            #plot the magitudes
            hist19 = plt.subplot(121)
            hist20 = plt.subplot(122)
            hist19.hist(mag_19)
            hist20.hist(mag_20)
            plt.show()

    finally:
        conn.close()

    return average

