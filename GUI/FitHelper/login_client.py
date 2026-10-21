import requests

SERVER_URL = "http://192.168.15.118:8000/api/login"

def login(user_id, password):
    try:
        response = requests.post(
            SERVER_URL,
            json={
                "userID": user_id,
                "password": password
            },
            timeout=5
        )

        response.raise_for_status()
        data = response.json()

        return data["loginResult"]

    except requests.RequestException as e:
        print(f"AIFit Server 통신 오류: {e}")
        return False
    except (ValueError, KeyError) as e:
        print(f"서버 응답 오류: {e}")
        return False