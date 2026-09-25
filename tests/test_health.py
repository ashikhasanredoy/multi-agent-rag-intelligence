def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    # Root serves HTML interface
    assert "Multi-Agent" in response.text
    assert "html" in response.headers.get("content-type", "")


def test_health_endpoints(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_database_health(client):
    response = client.get("/health/database")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
