## Sentence Classification

### Why not Deep-Learning and other Linear Classification methods: 
* We have very small amount of data and LSTMs are not good with small amount of data, Hence need to use Linear Classifier
* Specially dealing with sort of text classification problems, SVM performs better than naïve bayes, Decision Tree Classifier

### Steps:
1. Preprocessing 
    * Create one csv file which stores all files(particularly all rows from each file) from **[labeled_articles](https://github.com/varadbhogayata/Sentence-Classification/tree/master/SentenceCorpus/labeled_articles)** folder ([__PreprocessedCSV.csv__](https://github.com/varadbhogayata/Sentence-Classification/blob/master/preprocessedCSV.csv) will be generated after executing cell[1] contaning all rows
    ```python
   csv_file = "enter_your_path/preprocessedCSV.csv"  
   path = "enter_your_path/labeled_articles/*.txt"    # path of all txt files
   files=glob.glob(path)  

   # add all txt data into single csv file
   for f in files:
       with open(f, "r") as in_text:
           in_reader = csv.reader(in_text, delimiter = '\t')
           with open(csv_file, "a") as out_csv:
               out_writer = csv.writer(out_csv)
               for row in in_reader:
                   out_writer.writerow(row)
    ```
    * To remove __####Abstract__ and __####Introduction__ from original text files, convert csv file to pandas dataframe and remove those rows using pandas
    * Remove whitespace from the column **label** as some of the columns contain 'OWNX' and 'OWNX ', therefore consider this as two different classes.
    * Cleaning the data by using regex 
    * Collect stopwords from [__stopwords.txt__](https://github.com/varadbhogayata/Sentence-Classification/blob/master/SentenceCorpus/word_lists/stopwords.txt) file given in the data and store it in **stopword list** (Later will be used as input parameter to CountVectorizer)
    ```python
      # load stopwords from stopwords.txt
      file_stopwords = open("enter_your_path/word_lists/stopwords.txt","r")
      stopwords = []
      for w in file_stopwords:
          if w.endswith('\n'):
              w = w[:-1]
              stopwords.append(w)
    ```
    
2. TF-IDF features followed by SVM multiclass classification(oneVsOther)
    * Store sentences data from pandas dataframe to __X__ and labels to __y__ and convert them to numpy vectors
    * Split data into training and testing with 80:20 ratio and set any random seed value so that we can produce the same result with exact same accuracy and weights
    * Create a model having pipeline of CountVectorizer, TfidfTransformer, LinearSVC functions
    * Perform GridSearch to find optimum parameter for training the data
    * Provide the optimum parameter and train the model 
    * Plot Confusion matrix and accuracy score
    * Save the model so that it can be used directly again on this dataset
    * Test the model with inputing any sentence given from any files of [__unlabeled_articles__](https://github.com/varadbhogayata/Sentence-Classification/tree/master/SentenceCorpus/unlabeled_articles) folder 
    ![result](https://github.com/varadbhogayata/Sentence-Classification/blob/master/Images/Output%20Image.JPG)
