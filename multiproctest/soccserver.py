# Main Program
import socket
import json
import threading

data = {"variable1": 42, "variable2": "Hello"}

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9000))
    server.listen(1)
    while True:
        conn, addr = server.accept()
        conn.sendall(json.dumps(data).encode('utf-8'))
        conn.close()

threading.Thread(target=start_server, daemon=True).start()

# Your event loop here
while True:
    data["variable1"] += 1
