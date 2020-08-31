import joblib


val={1:'POSITIVE', 0:'NEGATIVE'}

model=joblib.load('Models/Sentiment-Model.pkl')
cv=joblib.load('Models/Vectorizer.pkl')


msg='ofcourse gay men dress well.They didnot spend all time in closet doing nothing'
msg=[msg]

test=cv.transform(msg)

index=(model.predict(test))[0]
print(val.get(index))