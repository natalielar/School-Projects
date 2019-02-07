# solutions.py
"""Volume 3: SQL 2.
<Name> Natalie Larsen
<Class> 001
<Date> 11-15-18
"""

import sqlite3 as sql
import numpy as np


# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #choose student name
            students = cur.execute("Select SI.StudentName "
                        #join table StudentInfo and Student grades on the student ID
                        "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                        "ON SI.StudentID = SG.StudentID "
                        #return the ones with B grades
                        "WHERE SG.Grade == 'B';").fetchall()
    finally:
        conn.close()

    #remove strings from tuples
    return [np.ravel(x) for x in students]


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Choose student name, major name and grade
            students = cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                        #from StudentInfo and MajorInfo where MajorID matches
                        "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                        "ON SI.MajorID == MI.MajorID "
                        #joined with StudentGrades where StudentID matches
                        "INNER JOIN StudentGrades AS SG "
                        "ON SI.StudentID == SG.StudentID "
                        #return where the course id is 1
                        "WHERE SG.CourseID == 1;").fetchall()

    finally:
        conn.close()
    return students


# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    classes = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Choose course name
            classes = cur.execute("SELECT CI.CourseName "
                        #from CourseInfo combined with StudentGrades where CourseID matches
                        "FROM CourseInfo AS CI INNER JOIN StudentGrades AS SG "
                        "ON CI.CourseID == SG.CourseID "
                        "GROUP BY SG.CourseID "
                        #where more than 5 people are in it
                        "HAVING COUNT(*) >= 5").fetchall()

    finally:
        conn.close()

    # remove strings from tuples
    return [np.ravel(x) for x in classes]


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    classes = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Choose Major name and number of students in it
            classes = cur.execute("SELECT MI.MajorName, COUNT(*) AS num_students "
                                  #from Student Info and Major Info on MajorID
                                  "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo as MI "
                                  "ON MI.MajorID == SI.MajorID "
                                  "GROUP BY MI.MajorName "
                                  #return the results in descending num of students, alphabetical order
                                  "ORDER BY num_students DESC, MI.MajorName ASC").fetchall()

    finally:
        conn.close()

    return classes


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Choose Student Name and Major Name
            students = cur.execute("SELECT SI.StudentName, MI.MajorName "
                                    #On StudentInfo with MajorInfo where MajorID matches
                                   "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo as MI "
                                   "ON MI.MajorId == SI.MajorID "
                                    #return where last name starts with C
                                   "WHERE SI.StudentName LIKE '% C%';").fetchall()

    finally:
        conn.close()

    return students



# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #choose Student name, number of classes and gpa
            students = cur.execute("SELECT StudentName, COUNT(*), AVG(gpa) "
                                   "FROM ( "
                                        #look at each students grades and classes
                                        "SELECT StudentName, CASE "
                                        #cases for grades and their gpa values
                                        "WHEN SG.Grade IS 'A+' THEN 4.0 "
                                        "WHEN SG.Grade IS 'A' THEN 4.0 "
                                        "WHEN SG.Grade IS 'A-' THEN 3.7 "
                                        "WHEN SG.Grade IS 'B+' THEN 3.4 "
                                        "WHEN SG.Grade IS 'B' THEN 3.0 "
                                        "WHEN SG.Grade IS 'B-' THEN 2.7 "
                                        "WHEN SG.Grade IS 'C+' THEN 2.4 "
                                        "WHEN SG.Grade IS 'C' THEN 2.0 "
                                        "WHEN SG.Grade IS 'C-' THEN 1.7 "
                                        "WHEN SG.Grade IS 'D+' THEN 1.4 "
                                        "WHEN SG.Grade IS 'D' THEN 1.0 "
                                        "WHEN SG.Grade IS 'D-' THEN 0.7 END AS gpa "
                                        #On StudentInfo with StudentGrades where StudentID matches
                                        "FROM StudentInfo AS SI LEFT OUTER JOIN StudentGrades AS SG "
                                        "ON SI.StudentID == SG.StudentID) "
                                   "GROUP BY StudentName "
                                    #return results in descending gpa
                                   "ORDER BY AVG(gpa) DESC;").fetchall()

    finally:
        conn.close()

    return students

