from cgitb import reset
from distutils.debug import DEBUG
from flask import Flask, render_template, request, redirect, url_for,flash
# from werkzeug.util import secure_filename
import os
from main import *
import time


UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    context = {}
    fpath = 'abc'
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(url_for('upload_file'))
            # return ('',404)

            # return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fpath)
            process_image(fpath)
            time.sleep(2)
            # rec,probb = main(fpath)
            # tf.compat.v1.reset_default_graph()

            # time.sleep(2)
            rec,prob = main('static\\temp\\img.jpg')
            # if prob1 >= probb:
            #      rec = rec1
            #      prob = prob1
            # else: 
            #     rec = recb
            #     prob = probb
            
            print(rec,prob)
            try:
                prob = prob.round(3)
            except:
                pass
            context = {'rec':rec, 'prob':prob}
            try:
                context['fpath'] = fpath
            except:
                pass
            print(context)
 

            # return redirect(url_for('upload_file'))
            # return redirect(url_for('upload_file', name=filename))
    return render_template('index.html', context = context)
    
    # return render_template('index.html', rec=rec, prob = str(prob), fpath=fpath)

    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''

# @app.route('/')
# def index():
#     rec,prob = main('..\\data\\alive.png')
    
#     # return f'rec is {rec}, prob is {prob}'
#     context = {rec:rec, prob:str(prob)}
#     print(rec,prob)

#     return render_template('index.html', context = context, rec=rec,prob=round(prob,2))


# @app.route('/', methods=['POST'])
# def upload_file():
#     uploaded_file = request.files['files']
#     if uploaded_file.filename != '':
#         uploaded_file.save(uploaded_file.filename)
#         print(uploaded_file.filename)
#     return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug = True)