import requests
import time
import json

def requestfromserver(
    script,
    scriptName="foo",
    algebra="cga",
    baseUrl = "https://gaalopweb.esa.informatik.tu-darmstadt.de/gaalopweb/res/",
    apimode="compile",
    codegen="de.gaalop.python.Plugin",
    debugprint=False,
    cache=None,
    overridecash=False
):
    script+=f"\n//{scriptName=} {algebra=} {baseUrl=} {apimode=} {codegen=}"#for cache,
    if cache is not None and not overridecash:
        #print("caching")
        if text:=cache.get(script):
            if debugprint:
                print("load from cache")
            return text
    
    if debugprint:
        print("GAALOPScript will be uploaded now! Please wait.")
    #algebra="..."
    ticketurl=f"{baseUrl}api/{apimode}"
    r = requests.post(url =ticketurl , data=script, params= {"InputScript": scriptName+".clu",  "CodeGenerator":codegen ,"AlgebraName":algebra }) 
    response = json.loads(r.content,strict=False)
    id=response["id"]
    waiturl  =f"{baseUrl}api/result/{id}"
    resulturl=f"{baseUrl}api/result/{id}/0"
    if debugprint:
        print(f"Compilation started using the ID {id}! Please wait for the result.")
        print(waiturl)
    # Wait on result compilationSucceeded or compilationException and check this each second
    
    while True:
        time.sleep(1)

        # Get current status of compilation
        response = json.loads(requests.get(url = waiturl).content,strict=False)
        status=response["status"]
        if debugprint:
            print(status)
        if   status== "compilationSucceeded":# if compilation has ended successful, end the loop
            break
        elif status == "compilationException":# if compilation has ended with errors, abort the function
            #if error is null wrong algebra
            raise Exception(f"Compilation was NOT successful. You can find an error description on:\n{baseUrl}python/result?id={id}\n{response['exception']}\n{response}")
        elif status=="noTaskWithThisID":
            raise Exception(f"Compilation was NOT successful.\nnoTaskWithThisID\n{response}")
    #print("Compilation was successful. You will find the generated code in the file: {destinationFilename}".format(destinationFilename=scriptName+".py"))
    if debugprint:
        print("Compilation was successful.")
    # Write result to destination file
    text=requests.get(url = resulturl).text
    if cache is not None:
        cache[script]=text
    return text

def servervisualizehtml(script,**kwargs):
    text=requestfromserver(script,apimode="visualize",**kwargs)
    with open("visualization.html.j2", "r") as template_file:
        return template_file.read().replace("{{ script }}",text)

def genpyfromgaalop(script,algebra="cga",**kwargs):
    return requestfromserver(script,algebra=algebra,apimode="compile",codegen="de.gaalop.python.Plugin",**kwargs)