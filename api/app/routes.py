# ------ routes.py ------

# Implementation for all api endpoints.

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from flask import Blueprint, request, jsonify
from .model_service import generate_res

bp = Blueprint('api', __name__)

@bp.route('/query', methods=['POST', 'OPTIONS'])
def query_model():
    """
    converts the user question to embedding
    and queries the model

    :return: response from model as a json
    """
    if request.method == 'OPTIONS':
        return '', 204 

    data = request.get_json()
    question = data.get('question', '')

    #check if question attr is given
    if not question:
        return jsonify({"error": "Question is required"}), 400

    response = generate_res(question)
    return jsonify({"answer": response})