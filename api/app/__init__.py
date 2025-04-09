# ------ __init__.py ------

# Backend Init.

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale


from flask import Flask
from flask_cors import CORS
from .routes import bp

#app registration
def create_app():
    app = Flask(__name__)
    CORS(app, origins=[
        "http://localhost:3000",
        "https://bentune-backend.onrender.com/" 
    ], supports_credentials=True)
    app.register_blueprint(bp)
    return app