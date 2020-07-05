# author : GOUTAM KUMAR GHADAI
import sys
import pandas as pd
import numpy as np
import nltk 
from sqlalchemy import create_engine
import sqlite3


def load_data(mess_filepath, catt_filepath):
    '''
    input: (
        mess_filepath: csv file 
        catt_filepath: csv file 
            )
    It reads in two csv files and outputs a merged dataframe df
    output: (pandas dataframe df)
    '''
    # loading the messages dataset to mess variable
    mess = pd.read_csv(mess_filepath)
    # load categories dataset
    catt = pd.read_csv(catt_filepath)
    # merge datasets
    df = mess.merge(catt, on='id')
    # return df 
    return df


def clean_data(df):
    '''
    input: (
        df: It is pandas dataframe 
            )
    It reads pandas dataframe and formats it for the ML model 
    output: (pandas dataframe df)
    '''
    #creating a dataframe of the 36 individual category columns and save it to catt variable
    catt = df.categories.str.split(';', expand=True)
    #plucking out first row to the row variable
    row = catt[:1]
    #extracting a list of new column names for categories.
    catt_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    #renaming the columns of `catt`
    catt.columns = catt_colnames

    # converting category values to just numbers 0 or 1
    for c in catt:
        # setting each value to be the last character of the string
        catt[c] = catt[c].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        # to convert column from string to numeric
        catt[c] = catt[c].astype(int)

    # dropping the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenating the original dataframe with the new `catt dataframe
    df = pd.concat((df, catt), axis=1)
    # dropping the duplicates
    df.drop_duplicates(inplace=True)
    # checking for the duplicates
    assert len(df[df.duplicated()]) == 0
    # return df
    return df


def save_data(df, db_filename):
    '''
    input: (
        df: It is a pandas dataframe 
        db_filename: database filename
            )
    It saves the pandas dataframe to database using create_engine
    output: Nothing
    '''
    tb_name = 'disaster'
    # creating a engine to store 
    engine = create_engine('sqlite:///{}'.format(db_filename))
    # saving dataframe to database
    df.to_sql(tb_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()