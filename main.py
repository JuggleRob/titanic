from flask import Flask
from flask_restful import Resource, Api
import pickle
import pandas as pd
from titanic_methods import get_age_group, get_sex

app = Flask(__name__)
api = Api(app)
model = pickle.load(open('classifier.pkl','rb'))

class MakePrediction(Resource):
    def get(self, Pclass, Sex, Age, FamilySize):
        input = pd.DataFrame({
            'Pclass': Pclass,
            'Sex': Sex,
            'AgeGroup': get_age_group(Age),
            'FamilySize': FamilySize
        }, index=[0,1,2,3])
        if model.predict(input)[0] == 0:
            return {
                'Survived prediction': 'False',
                'Certainty prediction': model.predict_proba(input)[0][0],
                'Sex': get_sex(Sex), 
                'Age': Age, 
                'Pclass': Pclass, 
                'FamilySize': FamilySize
            }
        else:
            return {
                'Survived prediction': 'True', 
                'Certainty prediction': model.predict_proba(input)[0][1],
                'Sex': get_sex(Sex), 
                'Age': Age, 
                'Pclass': Pclass, 
                'FamilySize': FamilySize
            }

class Help(Resource):
    def get(self):
        return {
            'url format': 'http://127.0.0.1:5000/<int:Sex>/<int:Age>/<int:Pclass>/<int:FamilySize>',
            'sex format': '0 = female , 1 = male',
            'age format': '0, 1, 2, 3, ...',
            'pclass format': '1, 2 or 3',
            'familysize format': '1, 2, 3, ... (note that family size is including yourself)'
        }
api.add_resource(MakePrediction, '/<int:Sex>/<int:Age>/<int:Pclass>/<int:FamilySize>')
api.add_resource(Help, '/')

if __name__ == '__main__':
    app.run(debug=False)