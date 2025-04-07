# ------ __init__.py ------

# Backend Init.

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale


from flask import Flask
from .routes import bp

#app registration
def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app