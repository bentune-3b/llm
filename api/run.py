# ------ run.py ------

# runs the backend architecture of bentune

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from flask import Flask
from app import create_app

if __name__ == '__main__':
    application = create_app()
    application.run()