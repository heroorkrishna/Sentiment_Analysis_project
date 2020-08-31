from flask import Flask,render_template,request,redirect
from flask_cors import CORS

import joblib

app=Flask(__name__ )
CORS(app)


val={1:'POSITIVE', 0:'NEGATIVE'}

model=joblib.load('Models/Sentiment-Model.pkl')
cv=joblib.load('Models/Vectorizer.pkl')

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/analyse/',methods=['POST'])
def analyse():
    print('hello')
    data=request.form['data']
    print(data)
    msg=[data]

    test = cv.transform(msg)
    index = (model.predict(test))[0]
    res=(val.get(index))

   #195e83
    
    if(index==1):

       return render_template('/result.html',value=res)

    else:

        return render_template('/neg.html', value=res)


if __name__ == '__main__':
    app.run()
