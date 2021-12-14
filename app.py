from flask import Flask, redirect, url_for, render_template, request
import prosesbackend
app = Flask(__name__, static_url_path = "/static", static_folder = "static")

@app.route('/')
def hello():
    return render_template('sentiment.html')

@app.route('/main',methods=['POST'])
def main():
    prosesbackend.main(request)
    return render_template('out.html')

if __name__=="__main__":
    app.run()