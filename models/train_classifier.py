import sys
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(db_filepath):
    '''
    input: (
        db_filepath: path to database
            )
    It loads data from the sqlite database 
    output: (
        X: X is the features dataframe
        y: y is the target dataframe
        catt_names: names of targets
        )
    '''
    tb_name = 'disaster'
    # loading the data from the database
    eng = create_engine('sqlite:///{}'.format(db_filepath))
    df = pd.read_sql_table(tb_name, eng)
    # create features X and target dataframe y
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    # get names 
    catt_names = y.columns
    return X, y, catt_names


def tokenize(text, lemma=True, rmv_stop=True, rmv_short=True):
    '''
    input: (
        text: 
        lemma: True if lemmatize word, 
        rmv_stop: Its value is True if remove stopwords  
        rmv_short: Its value is True if remove short words > 2
            )
    This function tokenizes the text into tokens separately that is given to it
    output: (
        tokens : It returns cleaned tokens in list 
            )
    '''
    # It lists out the stopword list 
    STOPWORDS = list(set(stopwords.words('english')))
    # initializing of the lemmatier to the variable lemmatizer
    lemmatizer = WordNetLemmatizer()
    # splitting up the string into words called tokens
    tokens = word_tokenize(text)
    # removeing the short words from the tokens list
    if rmv_short: tokens = [tok for tok in tokens if len(tok) > 2]
    # putting words into their base form from the tokens list
    if lemma: tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    # removing the stopwords from tokens list
    if rmv_stop: tokens = [tok for tok in tokens if tok not in STOPWORDS]
    # returning the cleaned list 
    return tokens


def build_model():
    '''Building of the classification model'''
    # Instatiating the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=100)))
    ])
    # Setting the hyper-parameter grid to search for the particular hyperparameter for the model
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto'],
        'clf__estimator__n_estimators': [100, 250],
    }
   
    # Intantiating the GridSearchCVmodel 
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=4, cv=3)
    
    # It returns the best hyperparameter based model
    return cv


def evaluate_model(model, X_test, Y_test, catt_names):
    '''
    input: (
        model: trained model 
        X_test: Test features 
        Y_test: Test labels 
        catt_names: names of lables
            )
    It evaluates a trained model against a test dataset that is given to it
    '''
    # getting predictions for the test dataset
    y_preds = model.predict(X_test)
    # printting out the classification report for the accuracy of the model
    print(classification_report(y_preds, Y_test.values, target_names=catt_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_preds)))


def save_model(model, modd_filepath):
    '''
    input: (
        model: trained model which needs to be saved 
        modd_filepath: It is the filepath to save model in binary form 
            )
    Save the model to a Python pickle file 
    '''
    # saveing the model binary into the given path
    pickle.dump(model, open(modd_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()