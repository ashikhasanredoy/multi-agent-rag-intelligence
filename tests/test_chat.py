from unittest.mock import patch, AsyncMock


def test_chat_endpoint_mocked(client):
    with patch("app.api.chat.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Hello! I am ready to help you."

        payload = {
            "message": "Hello"
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert data["answer"] == "Hello! I am ready to help you."
        assert data["agent"] == "general_agent"
        assert isinstance(data["sources"], list)
        assert data["confidence"] == 1.0
        assert "latency_seconds" in data
