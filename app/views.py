from app import app

from app import pd
from app import np

from flask import render_template, request
from jinja2 import escape
from flask_bootstrap import Bootstrap

""" Call function from a python file, after all the import"""
from app.preprocess import PreprocessInput
from app.preprocess import ElmoInput
from app.preprocess import CosSimi
from app.preprocess import SelectKeyword
from app.preprocess import FinalResult


@app.route('/', methods=['GET','POST'])
def index():
    area_input = ''
    keyword = []
    area = []
    area_input = ''
    keyword_input = ''
    nb_result = 10
    out_blurb = []
    out_title = []
    out_author = []
    out_isbn = []



    data_vect = np.load('/home/Mathieu23IA/mysite/app/data/bv.npy')
    data_csv = pd.read_csv('/home/Mathieu23IA/mysite/app/data/books_clean.csv')
    data_csv_display = pd.read_csv('/home/Mathieu23IA/mysite/app/data/books_clean_display.csv')
    size_data = 54649
    """len(data_csv)-1"""
    


    if request.method == "POST":
        area_input = escape(request.form['abstract_input'])
        keyword_input = escape(request.form['keyword_input'])
        area, keyword = PreprocessInput(area_input, keyword_input)

        if (area_input != '' and area_input != ' '):
            area_vect = ElmoInput(area)
            similarite = CosSimi(nb_result, area_vect, data_vect) 
            if (keyword != ''):
                final_word = SelectKeyword(data_csv, keyword)
            else:
                final_word = ''

            out_blurb, out_title, out_isbn, out_author, score_display = FinalResult(final_word, similarite, data_csv, data_csv_display, size_data, nb_result)

            
        else:
            error_area = "Wrong format of data"

    
    return render_template("public/display_result.html", **locals())