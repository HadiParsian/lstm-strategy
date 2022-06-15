import numpy as np
import pandas as pd

from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


import warnings
warnings.filterwarnings('ignore')
from bokeh.plotting import figure, show, output_file, output_notebook, save
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar
from datetime import datetime

from lstm import flask_html




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method=='POST':  
        if request.form['content']!='':
            stock_name = request.form['content'].upper()
            flask_html(stock_name)
            return render_template('line_chart.html')
        else: 
            return render_template('index.html')
    else: 
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)