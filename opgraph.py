class OpNode:
    """
    Base class for all operation nodes in the computation graph.
    """
    def __init__(self, parents):
        self.parents = [OpNode.nodify(p) for p in parents]
        self.children = []
        for parent in self.parents:
            parent.children.append(self)

    def addparent(self,p):
        self.parents.append(p)
        for parent in self.parents:
            parent.children.append(self)


    @staticmethod
    def nodify(value):
        """
        Converts a value into an appropriate node.
        Supports:
        - `OpNode` objects (returned as-is)
        - Numeric values (converted to `ConstNode`)
        """
        if isinstance(value, OpNode):
            return value
        if isinstance(value, (int, float)):
            return ConstNode(value)
        raise TypeError(f"Unsupported type for nodify: {type(value)}")
    
    def copy_graph(self,mapping=None,copydown=True):
        if mapping is None:
            mapping=dict()

        if self in mapping:
            return mapping[self]
        new_node=mapping[self]=self._copy_op()

        # Recursively copy parents
        new_node.parents = [parent.copy_graph(mapping,copydown) for parent in self.parents]

        for p in new_node.parents:
            p.children.append(new_node)
        
        if copydown:
            #  copy the children of this node
            for child in self.children:
                child.copy_graph(mapping,copydown)


        return new_node

    def topological_sort(self, visited=None, ordering=None):
        if visited is None:visited=set()
        if ordering is None:ordering=[]
        if self in visited:return ordering
        visited.add(self)
        for p in self.parents:
            p.topological_sort(visited, ordering)
        ordering.append(self)

        return ordering
    
    def asplanstr(self):
        nodes=self.topological_sort()
        nodenames=dict()
        nodestr=[]
        for i,n in enumerate(nodes):
            name=f"node{i}"
            #name=f"n{id(n)}"
            nodenames[n]=name
            nodestr.append(name+"="+n.nodedef(nodenames[p]for p in n.parents))
        return "\n".join(nodestr)
        


    # Operator overloading
    def __add__(self, other):
        """Overload the + operator."""
        return AddNode([self, other])

    def __mul__(self, other):
        """Overload the * operator."""
        return MulNode([self, other])

    def __sub__(self, other):
        """Overload the - operator."""
        return SubNode([self, other])

    def __truediv__(self, other):
        """Overload the / operator."""
        return DivNode([self, other])

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.parents))})"

    def nodedef(self,parentnames):
        return f"{self.__class__.__name__}({', '.join(parentnames)})"
    
    def _copy_op(self):
        return type(self)([])
    def signature(self,associative=False,args=None):
        #sort,parents,type
        if args is not None:
            typearg=(type(self).__name__,args)
        else:
            typearg=type(self).__name__
        if associative:
            parents=sorted(self.parents,key=id)
        else:
            parents=self.parents
        return tuple(parents),typearg


class ConstNode(OpNode):
    """
    Represents a constant value in the computation graph.
    """
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def __repr__(self):
        return f"Const({self.value})"
    def nodedef(self,parentnames):
        return self.__repr__()
    def _copy_op(self):
        return ConstNode(self.value)
    def signature(self):
        return super().signature(args=self.value)


class VarNode(OpNode):
    """
    Represents a variable in the computation graph.
    """
    def __init__(self, varname):
        super().__init__([])
        self.varname = varname

    def __repr__(self):
        return f"Var({self.varname})"
    def nodedef(self,parentnames):
        return self.__repr__()
    def _copy_op(self):
        return VarNode(self.varname)
    def signature(self):
        return super().signature(args=self.varname)


# Operation-Specific Nodes
class AddNode(OpNode):
    """
    Represents an addition operation in the computation graph.
    """

    def __repr__(self):
        return f"Add({self.parents[0]}, {self.parents[1]})"
    def signature(self):
        return super().signature(associative=True)
class EndpointNode(OpNode):
    """
    Represents an addition operation in the computation graph.
    """
    def subexpressionelimination(self):
        nodes=self.topological_sort()
        sigtonode=dict()#repr->node
        def nodesignature(node:OpNode):
            parents,args=node.signature()
            return tuple(id(p)for p in parents),args
        

        for n in nodes:
            sig=n.signature()
            samenode=sigtonode.get(sig,None)
            if samenode is not None:#if the nodes
                samenode.children.extend(n.children)
                for c in n.children:
                    c.parents= [samenode if p is n else p for p in c.parents]
                n.children.clear()
                n.parents.clear()
            else:
                sigtonode[sig] = n
            #print(list(sigtonode.values())[-1].asplanstr())
        return list(sigtonode.values())
    
   





    #def __repr__(self):
    #    return f"Add({self.parents[0]}, {self.parents[1]})"

class MulNode(OpNode):
    """
    Represents a multiplication operation in the computation graph.
    """

    def __repr__(self):
        return f"Mul({self.parents[0]}, {self.parents[1]})"
    def signature(self):
        return super().signature(associative=True)


class SubNode(OpNode):
    """
    Represents a subtraction operation in the computation graph.
    """

    def __repr__(self):
        return f"Sub({self.parents[0]}, {self.parents[1]})"


class DivNode(OpNode):
    """
    Represents a division operation in the computation graph.
    """
    def __repr__(self):
        return f"Div({self.parents[0]}, {self.parents[1]})"


# Usage Example
x = VarNode("x")

def fib(n):
    if n==1 or n==2:return ConstNode(1)
    return fib(n-1)+fib(n-2)

c=fib(30)
e=EndpointNode([c]).copy_graph(copydown=False)
#print(e.asplanstr())
print(len(e.topological_sort()))
e.subexpressionelimination()
print(e.asplanstr())

