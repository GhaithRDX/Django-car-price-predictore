from django.shortcuts import render
from .car import w_calc,prepare_X
import pandas as pd
import numpy as np

w_0 , w = w_calc()

def predict(request):
    
    if request.method =='POST':
     
        df ={
            'make': request.POST['make'],
            'model': request.POST['model'],
            'year': int(request.POST['year']),
            'engine_fuel_type': request.POST['engine_fuel_type'],
            'engine_hp': int(request.POST['engine_hp']),
            'engine_cylinders': int(request.POST['engine_cylinders']),
            'transmission_type': request.POST['transmission_type'],
            'driven_wheels': request.POST['driven_wheels'],
            'number_of_doors': int(request.POST['number_of_doors']),
            'market_category': request.POST['market_category'],
            'vehicle_size': request.POST['vehicle_size'],
            'vehicle_style': request.POST['vehicle_style'],
            'highway_mpg': int(request.POST['highway_mpg']),
            'city_mpg': int(request.POST['city_mpg']),
            'popularity':int(request.POST['popularity']) ,
            
        }
        
        X_test = prepare_X(pd.DataFrame([df]))
        y_pred = w_0 + X_test.dot(w)
        price  = np.expm1(y_pred)[0].astype(int)
        
        context = {
            'price':price ,
        }
        
        return render(request , 'index.html',context)

    else:
        
        return render(request , 'index.html')
