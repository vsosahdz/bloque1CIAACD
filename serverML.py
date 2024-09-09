#Python Libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
# Files management
import os
from werkzeug.utils import secure_filename

# Load model
dt=joblib.load('dt1.joblib')
# Create Flask app
server = Flask(__name__)

# Define a route to send JSON data
@server.route('/predictjson', methods=['POST'])
def predictjson():
    #Procesar datos de entrada
    data = request.json
    print(data)
    inputData = np.array([
        data['pH'],
        data['sulphates'],
        data['alcohol']
    ])
    #Predicir utilizando la entrada y el modelo
    result = dt.predict(inputData.reshape(1, -1))
    #Enviar respuesta
    return jsonify({'Prediction': str(result[0])})

#forms
@server.route("/formulario",methods=['GET'])
def formulario():
    return render_template('pagina.html')

#Envio de datos a través de Archivos
@server.route('/modeloFile', methods=['POST'])
def modeloFile():
    f = request.files['file']
    filename=secure_filename(f.filename)
    path=os.path.join(os.getcwd(),'files',filename)
    f.save(path)
    file = open(path, "r")
    
    for x in file:
        info=x.split()
    print(info)
    datosEntrada = np.array([
            float(info[0]),
            float(info[1]),
            float(info[2])
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de Forms
@server.route('/modeloForm', methods=['POST'])
def modeloForm():
    #Procesar datos de entrada 
    contenido = request.form
    
    datosEntrada = np.array([
            contenido['pH'],
            contenido['sulphates'],
            contenido['alcohol']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

if __name__ == '__main__':
    server.run(debug=False,host='0.0.0.0',port=8080)