## Sentence Classification

### Why not Deep-learning: 
    We have very small amount of data and LSTMs are not good with small amount of data, Hence need to use Linear Classifier

### Steps:
1. Preprocessing 
    * Create one csv file which stores all files(particularly all rows from each file) from **labeled_articles** folder (__PreprocessedCSV.csv__ will be generated after executing cell[1] contaning all rows     
    * To remove __####Abstract__ and __####Introduction__ from original text files, convert csv file to pandas dataframe and remove those rows using pandas
    * Remove whitespace from the column **label** as some of the columns contain 'OWNX' and 'OWNX ', therefore consider this as two different classes.
    * Cleaning the data by using regex 
    * Collect stopwords from stopwords.txt file given in the data and store it in **stopword list** (Later will be used as input parameter to CountVectorizer)
    
2. Tfidf features followed by SVM multiclass classification(oneVsOther)
    * Store sentences data from pandas dataframe to __X__ and labels to __y__ and convert them to numpy vectors
    * Split data into training and testing with 80:20 ratio and set any random seed value so that we can produce the same result with exact same accuracy and weights
    * Create a model having pipeline of CountVectorizer, TfidfTransformer, LinearSVC functions
    * Perform GridSearch to find optimum parameter for training the data
    * Provide the optimum parameter and train the model 
    * Plot Confusion matrix and accuracy score
    * Save the model so that it can be used directly again on this dataset
    * Test the model with inputing any sentence given from any files of **unlabeled_articles** folder 
