from flask import Flask, render_template
from flask_restful import Api
from Controller.fileSend import DicomSend
from Controller.otsuthreshold import OtsuThreshold
from Controller.gammaCorrection import GammaCorrection
from Controller.contrastAdjust import ContrastAdjust
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

@app.route('/')
def index():
    return render_template('page_1/index.html')

api.add_resource(DicomSend,'/dicomFile/metadata')
api.add_resource(GammaCorrection,'/gammaCorrection/<float:gamma>')
api.add_resource(OtsuThreshold,'/otsuThreshold/<int:max>')
api.add_resource(ContrastAdjust,'/contrastAdjust/<int:contrast>/<int:brightness>')

if __name__ == '__main__':
    app.run(port=5000, debug=True)