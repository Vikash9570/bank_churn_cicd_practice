from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            RowNumber=float(request.form.get('RowNumber')),
            CustomerId = float(request.form.get('CustomerId')),
            Surname = float(request.form.get('Surname')),
            CreditScore = float(request.form.get('CreditScore')),
            Geography = float(request.form.get('Geography')),
            Gender = float(request.form.get('Gender')),
            Age = request.form.get('Age'),
            NumOfProducts= request.form.get('NumOfProducts'),
            IsActiveMember = request.form.get('IsActiveMember'),
            HasCrCard = request.form.get('HasCrCard'),
            EstimatedSalary= request.form.get('EstimatedSalary'),
            Exited = request.form.get('Exited')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)



if __name__=="__main__":
    app.run(host='127.0.0.1',debug=True)