from django.shortcuts import render
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__),'../savedmodel/model.joblib')

try:
    print(f"'loading model from: {model_path}")
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def home(request):
    prediction = None

    if request.method == 'POST':
        try:
            iq = float(request.POST.get('iq'))
            cgpa = float(request.POST.get('cgpa'))
            user_input = [[iq,cgpa]]

            if model:
                result = model.predict(user_input)[0]
                prediction = round(result,3)
            else:
                prediction = "Error!! Model not found" 
        except Exception as e:
            prediction = f"Error! {e}"           

    return render(request,'index.html',{'result': prediction})


