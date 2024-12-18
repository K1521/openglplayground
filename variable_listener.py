from multiprocessing.connection import Listener, Client as ConnClient
import threading
import atexit
# Server class
class Server:
    def __init__(self, address=('localhost', 6000), authkey=b'secret password'):
        self.thread = threading.Thread(target=self._start_server, args=(address, authkey), daemon=True)
        self.thread.start()
        self.variableupdates=0


    def _start_server(self, address, authkey):
        import __main__ 
        mainglobals=__main__.__dict__
        #print(globals())
        listener = Listener(address, authkey=authkey)
        atexit.register(lambda:listener.close())
        print(f"Server listening on {address}")
        while True:
            conn = listener.accept()
            print('Connection accepted from', listener.last_accepted)
            while True:
                msg = conn.recv()
                match msg:
                    case ("getvariable", varname):
                        # Check if the variable exists in globals
                        try:
                            varnames=varname.split(".")
                            value=mainglobals[varnames.pop(0)]
                            for v in varnames:
                                value=value.__getattribute__(v)
                        except Exception as e:
                            conn.send(("getvariableanswer", False, varname, "Variable not found"))
                        else:
                            conn.send(("getvariableanswer", True, varname,value))
                    case ("setvariable", varname, value):
                        # Set the variable in globals
                        self.variableupdates+=1
                        mainglobals[varname] = value
                        conn.send(("setvariableanswer", True, varname, "Variable set successfully"))
                    # case ("getglobals",):
                    #     conn.send(("getglobalsanswer", True, mainglobals))
                    case ("close",):
                        # Close the server
                        print("Server shutting down")
                        conn.close()
                        break

# Client class
class Client:
    def __init__(self, address=('localhost', 6000), authkey=b'secret password'):
        self.conn = ConnClient(address, authkey=authkey)
        atexit.register(self.close)
    # def getglobals(self):
    #     # Send a "getvariable" request to the server
    #     self.conn.send(('getglobals',))
    #     # Wait for the server's response
    #     response = self.conn.recv()
    #     if response[0] == 'getglobalsanswer':
    #         success, value = response[1:]
    #         if success:
    #             return value
    #         else:
    #             raise KeyError(value)
    #     else:
    #         raise RuntimeError("Unexpected response from server")

    def __getitem__(self, key):
        # Send a "getvariable" request to the server
        self.conn.send(('getvariable', key))
        # Wait for the server's response
        response = self.conn.recv()
        if response[0] == 'getvariableanswer':
            success, varname, value = response[1:]
            if success:
                return value
            else:
                raise KeyError(f"Variable '{varname}' not found on the server")
        else:
            raise RuntimeError("Unexpected response from server")

    def __setitem__(self, key, value):
        # Send a "setvariable" request to the server
        self.conn.send(('setvariable', key, value))
        # Wait for the server's acknowledgment
        response = self.conn.recv()
        if response[0] == 'setvariableanswer':
            success, varname, message = response[1:]
            if success:
                return
            else:
                raise RuntimeError(f"Failed to set variable '{varname}': {message}")
        else:
            raise RuntimeError("Unexpected response from server")

    def close(self):
        # Send a close request to the server
        self.conn.send(("close",))
        self.conn.close()

# Example Usage
if __name__ == "__main__":
    # Start the server
    server = Server()

    # Create a client connection
    client = Client()

    # Set a variable remotely
    client['example_var'] = 123
    print("Set 'example_var' to 123")

    # Get the variable remotely
    try:
        value = client['example_var']
        print("Value of 'example_var':", value)
    except KeyError as e:
        print(e)

    # Attempt to get a non-existent variable
    try:
        value = client['non_existent_var']
        print("Value of 'non_existent_var':", value)
    except KeyError as e:
        print(e)

    # Shut down the server
    client.close()
