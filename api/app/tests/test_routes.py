import json
from app import create_app

def test_query_endpoint():
    """
    tests the query endpoint
    assert answer and data in the answer
    """

    #create an app instance
    app = create_app()
    client = app.test_client()

    #send post request
    response = client.post('/query',
        data=json.dumps({"question": "What is the capital of France?"}),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data
    assert "France" in data["answer"]