
# Wine Classifier WebApp
This is a 3 stage individual project developed by which uses a Random Forest classifier, Machine Learning Algorithm to train upon prexisting data about wine quality derived from kaggle and through feature scaling and PCA we improve our data quality once 
that is achieved we attempt to use better hyperparamters and decide the forest dept upon iterations ending up on 83. % accuracy level for the training and 81% for the test data repectively.
Once this is done we then go ahead to develop a GUI on the local machine using the streamlit library and deploying out project online through heroku.


## Acknowledgements

 - [Streamlit Documentation](https://docs.streamlit.io/)
 - [Hyperparameter tuning though GridSearchCV](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Authors

- [@SiddhantJha](https://github.com/SiddhantJha317)



## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



## Demo

https://sidwineapp.herokuapp.com/

![alt](https://github.com/SiddhantJha317/Wine_Quality_WebApp/blob/main/Screenshot%202022-08-26%20112158.png?raw=True)


## Model Construction

We go through the steps of algorithmic construction by using Scikit learn and pandas to firt look through the data at hand :

```python
data=pd.read_csv("winequality-red.csv")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])
```
The Code above reads in the csv file of wine classifcation as extracted from kaggle , we first use the pandas library to read in all the data and then separate them into X and y variables as y being the target variable and X being the features , then we go ahead and change the data types from pandas data frame to a numpy array so that sklearn can read the data in.

The next step would be to create  a scikit learn pipeline since entering the data each time would require rescaling , we use the standard scaler to scale down the data so that better sccuracy can be achieved while training.

```python
pipeline=make_pipeline(preprocessing.StandardScaler(),
                        RandomForestClassifier(n_estimators=100))
```
In the above step we also choose the classifcation algorithm to use , RandomForestClassifier being the choice with esmiators going a max dept of 100 trees.

To achieve better results on our algorithm since we suffer from data deficiency we attempt hyperparamter tuning on the RandomForestClassifier algorithm and then use the chosen paramters in the pipeline in use.

```python
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestclassifier__max_depth' : [None, 10, 7, 5, 3, 1]}  
```
Then we go ahead with the GridSearchCV class to implement these paramters on the default algorithm and train it on the whole dataset at once.

```python
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X, Y)
```
finally saving the trained model in a pkl file 
```python
model.save('model_pkl','w')
```
## Making the App Gui 
To make the App Gui and deploy the model quickly we use a python library called *streamlit*, through this we first construct a functioning Ui with sliders and then take those slider inputs as test features in model to predict a label output.

We first define a function that creates slders inside a container and then deploy the values from each shift in any of the sliders in a dictionary through a function.

```python
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.0, 16.0, 12.6)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.10,1.6 , 0.34)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.34 , 0.04)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,280.0 , 31.4)
        alcohol=st.sidebar.slider('alcohol', 8.42,15.0, 8.9)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
```
The above function takes in the data from sliders at change in the instance , with limitations upon the slider variation to avoid "illegal" inputs and then turns these inputs into dictionary targets with their pointers being the respective feature name as observed in the original dataset.

Now to use the dictionary we first convert the features returned by the function to a pandas DataFrame.
```python
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()
```
finally we use the same pandas DataFrame to feed into the x and y arrays and then use those arrays to predict labels to display as output with their respective probabilities.
```python
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
```
## Deployment

To deploy this streamlit project on heroku we must add three files to the repository , *requirements.txt* ,*setup.sh* and a heroku Procfile.

To create the requirements we can go the command line install pipreqs and then run  pipreqs on your local directory where *app.py* is stored it should give a requirements text file as output in the same folder.
```
>pip install pipreqs
>directory name
>pipreqs directory name

```
after this is done create a new file inside the same folder and name it setup.txt and then enter the following bash script.

```
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

```
once done change extension from *.txt* to *.sh* the above code would create a request instance to heroku when running the file on the web.

then Go ahead and create another txt file with the name *Procfile* and enter the following command.
 ```
 web: sh setup.sh && streamlit run app.py

 ```
Once done this change remove the txt extension and don't add any other.

Now we can push the App to the heroku server run the following codes.

```
> git add .
> git commit -m " final push"
> heroku login
> heroku create -a appname
> git push heroku main
```

## Run Locally

Clone the project

```bash
  git clone https://github.com/SiddhantJha317/Wine_Quality_WebApp
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install requirements.txt
```

Start the server

```bash
  streamlit run app.py
```

## Thanks
