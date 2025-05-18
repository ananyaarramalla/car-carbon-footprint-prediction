from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from model import db, TokenBlocklist
from auth import auth
from predict import predict
from datetime import timedelta
from flask import Flask, redirect, url_for, render_template

app = Flask(__name__, static_url_path='/static')
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
app.config['JWT_SECRET_KEY'] = 'sxahb2emqGHS'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

db.init_app(app)
jwt = JWTManager(app)

# Token blocklist check
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return TokenBlocklist.query.filter_by(jti=jwt_payload["jti"]).first() is not None

# Register blueprints
app.register_blueprint(auth, url_prefix='/auth')
app.register_blueprint(predict, url_prefix='/ml')

@app.route('/')
def home():
    return render_template("home.html")  # Make sure home.html exists

# Create tables
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

