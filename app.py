from flask import Flask
from flask_restful import Api
from Controller.otsuthreshold import OtsuThreshold
from Controller.gammaCorrection import GammaCorrection
from Controller.contrastAdjust import ContrastAdjust
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

api.add_resource(GammaCorrection,'/gammaCorrection/<float:gamma>')
api.add_resource(OtsuThreshold,'/otsuThreshold/<int:max>')
api.add_resource(ContrastAdjust,'/contrastAdjust/<int:contrast>/<int:brightness>')

if __name__ == '__main__':
    app.run(port=5000, debug=True)