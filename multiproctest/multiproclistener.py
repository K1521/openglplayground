from threading import Thread


a=1
b=2
c=3
def printglobals():
    d=4
    print(globals())

th=Thread(target=printglobals,daemon=True).start() 
print(1|2)