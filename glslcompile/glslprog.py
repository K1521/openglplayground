

class glslprogrammpart:
    def __init__(self,header=None,body=None,headerpath=None,bodypath=None,headerprio=0,bodyprio=0,name=None):
        if header is not None and headerpath is not None:
            raise ValueError("you supplied headerpath and header")
        if body is not None and bodypath is not None:
            raise ValueError("you supplied bodypath and body")
        if headerpath is not None:
            with open(headerpath) as file:
                header=file.read()
        if bodypath is not None:
            with open(bodypath) as file:
                body=file.read()
        self.header=header
        self.body=body
        self.headerprio=headerprio
        self.bodyprio=bodyprio
        self.name=name
class glslprogramm:
    def __init__(self,version=None):
        self.parts: list[glslprogrammpart]=[] 
        self.version=version           
    def __str__(self):
        strparts=[]
        if self.version is not None:
            version=str(self.version).removeprefix("#").lstrip(" ").removeprefix("version").lstrip(" ")
            strparts.append(f"#version {version}")

        names=dict()
        for i,part in enumerate(self.parts):
            if part.name is None:
                name=f"part{i}"
            else:
                name=part.name
            names[part]=(f"/*source={name}*/")


        headers=sorted(self.parts,key=lambda part:part.headerprio,reverse=True)
        for part in headers:
            if part.header is None:
                continue
            strparts.append(names[part])
            strparts.append(part.header)
        
        bodys=sorted(self.parts,key=lambda part:part.bodyprio,reverse=True)
        for part in bodys:
            if part.body is None:
                continue
            strparts.append(names[part])
            strparts.append(part.body)
        
        return "\n".join(strparts)