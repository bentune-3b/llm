# run.py
# ------------------------
# Runs the backend architecture of bentune.
# Use "0.0.0.0" for prod and default ports for local runs
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from flask import Flask
from app import create_app

if __name__ == '__main__':
    application = create_app()
    application.run("0.0.0.0", port=5000)
    # application.run()