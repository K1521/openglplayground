from collections import defaultdict

calls=0
class OpNode:
    """
    Base class for all operation nodes in the computation graph.
    """
    def __init__(self, parents):
        self.parents = [OpNode.nodify(p) for p in parents]
       

    @staticmethod
    def nodify(value)->"OpNode":
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
    
    def copy_graph(self,mapping=None):
        if mapping is None:
            mapping=dict()

        if self in mapping:
            return mapping[self]
        new_node=mapping[self]=self._copy_op()

        # Recursively copy parents
        new_node.parents = [parent.copy_graph(mapping) for parent in self.parents]

        return new_node

    # def topological_sortr(self, visited, ordering):
    #     if self in visited:
    #         return ordering
    #     visited.add(self)
    #     for p in self.parents:
    #         p.topological_sortr(visited, ordering)
    #     ordering.append(self)
    # def topological_sort(self, visited=None, ordering=None):
    #     if visited is None:visited=set()
    #     if ordering is None:ordering=[]
    #     self.topological_sortr(visited, ordering)
    #     return ordering
    
    
    def topological_sort(self, visited=None, ordering=None)->list["OpNode"]:
        if visited is None:visited=set()
        if ordering is None:ordering=[]
        if self in visited:
            return ordering
        visited.add(self)
        for p in self.parents:
            p.topological_sort(visited, ordering)
        ordering.append(self)
        return ordering
    def maxdepth(self, depths=None):
        if depths is None:depths=dict()

        d=depths.get(self,None)
        if d is not None:return d

        d=max((p.maxdepth()for p in self.parents),default=-1)+1

        depths[self]=d
        return d

        

        
    def topsortwithchildren(self):
        nodes=self.topological_sort()
        children={n:set() for n in nodes}
        for n in nodes:
            for p in n.parents:
                children[p].add(n)
        return nodes,children
    def childcount(self):
        nodes=self.topological_sort()
        children={n:0 for n in nodes}
        for n in nodes:
            for p in n.parents:
                children[p]+=1
        return children
    
    def asplanstr(self,compact=False):
        nodes=self.topological_sort()
        nodenames=dict()
        nodestr=[]
        for i,n in enumerate(nodes):
            name=f"node{i}"
            #name=f"n{id(n)}"
            nodenames[n]=name
            nodestr.append(name+"="+n.nodedef((nodenames[p]for p in n.parents),compact))
        return "\n".join(nodestr)
    def asplanstrcompact(self,compact=False):
        nodes=self.topological_sort()
        mergables=set(k for k,v in self.childcount().items() if v==1 )#nodes with 1 child
        nodenames=dict()
        nodestr=[]
        for i,n in enumerate(nodes):
            if n in mergables:
                nodenames[n]=n.nodedef((nodenames[p]for p in n.parents),compact)
            else:
                name=f"n{i}" if compact else f"node{i}"
                nodenames[n]=name
                nodestr.append(name+"="+n.nodedef((nodenames[p]for p in n.parents),compact))
        return "\n".join(nodestr)
    
    def backpropergation(self,variables=None):  
        if not isinstance(variables,list):
            return self.backpropergation([variables])[0]
        zero=ConstNode(0)
        doutdnode=defaultdict(lambda:zero)
        doutdnode[self]=ConstNode(1)#set derivative for output node
        topsort=self.topological_sort()
        for node in reversed(topsort):
            node.backpropergationnode(doutdnode)#set derivative for other nodes
        if variables is None:
            return doutdnode
        
        #if the tree has different variable objects wich are the same variable combine them
        sigtoderivs=defaultdict(set)
        for n in topsort:
            if isinstance(n,VarNode):
                sigtoderivs[n.signature()].add(doutdnode[n])
        def nodesum(nodes):
            if len(nodes)==0:return zero
            if len(nodes)==1:return next(iter(nodes))
            return AddNode(nodes)

        return [nodesum(sigtoderivs[v.signature()])for v in variables]
        

        


    # Operator overloading
    def __add__(self, other):
        """Overload the + operator."""
        return AddNode([self, other])
    def __radd__(self, other):
        return AddNode([other, self])

    def __mul__(self, other):
        """Overload the * operator."""
        return MulNode([self, other])
    def __rmul__(self, other):
        return MulNode([other,self])

    def __sub__(self, other):
        return SubNode([self, other])
    def __rsub__(self, other):
        """Overload the - operator."""
        return SubNode([other,self])

    def __truediv__(self, other):
        """Overload the / operator."""
        return DivNode([self, other])
    
    def __neg__(self):
        return NegNode([self])
    def __abs__(self):
        return AbsNode([self])
    def sqrt(self):
        return SqrtNode([self])

    def __repr__(self):
        return f"{self.__class__.__name__.removesuffix('Node')}({', '.join(map(str, self.parents))})"

    def nodedef(self,parentnames,small=False):
        return f"{self.__class__.__name__.removesuffix('Node')}({', '.join(parentnames)})"
    
    
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
    def consteliminationnode(self):
        return self
    
    
    def normalizenode(self):
        return self

    def replacenode(self,f,nodesigcash=None,nodecash=None,issimplified=False):
        if nodesigcash is None:nodesigcash=dict()#node.signature()->node
        if nodecash is None:nodecash=dict()


        cashednode=nodecash.get(self,None)
        if cashednode is not None:return cashednode

        self.parents=[p.replacenode(f,nodesigcash,nodecash,issimplified)  for p in self.parents]
        sig=self.signature()
        cashednode=nodesigcash.get(sig,None)
        if cashednode is not None:return cashednode

        if issimplified:
            simplified=self
        else:
            simplified=f(self)
            if simplified!=self:
                simplified=simplified.replacenode(f,nodesigcash,nodecash,issimplified=True)


        nodecash[self]=simplified
        nodesigcash[sig]=simplified

        return simplified
    def backpropergationnode(self,doutdnode):
        raise NotImplementedError(f"backpropergation not implemented in nodetype {type(self)}")



class EndpointNode(OpNode):
    """
    Represents an addition operation in the computation graph.
    """
    def subexpressionelimination(self):#TODO speedtest
        nodes,children=self.topsortwithchildren()

        sigtonode=dict()#repr->node
        for n in nodes:
            sig=n.signature()
            samenode=sigtonode.get(sig,None)
            if samenode is not None:
                for c in children[n]:
                    c.parents= [samenode if p is n else p for p in c.parents]
            else:
                sigtonode[sig] = n
            #print(list(sigtonode.values())[-1].asplanstr())
        return list(sigtonode.values())
    
    def mergenodes(self):
        nodes,children=self.topsortwithchildren()
        mergables=set(k for k,v in children.items() if len(v)==1 )
        
        def nodemerger(node):
            for Nodetype in [AddNode,MulNode]:
                if isinstance(node,Nodetype):
                    newparents=[]
                    mergedone=False
                    for p in node.parents:
                        if isinstance(p,Nodetype) and p in mergables:
                            newparents.extend(p.parents)
                            mergedone=True
                        else:
                            newparents.append(p)
                    if mergedone==False:return node
                    newnode=Nodetype(newparents) 
                    if node in mergables:
                        mergables.add(newnode)
                    return newnode
            return node
        self.replacenode(nodemerger)
    def backpropergation(self,variables=None):
        if len(self.parents)!=1:
            raise NotImplementedError("currently backpropergation is only supported for one output")
        return self.parents[0].backpropergation(variables)
    






class ConstNode(OpNode):
    """
    Represents a constant value in the computation graph.
    """
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def __repr__(self):
        return f"Const({self.value})"
    def nodedef(self,parentnames,small=False):
        if small:return str(self.value)
        return self.__repr__()
    def _copy_op(self):
        return ConstNode(self.value)
    def signature(self):
        return super().signature(args=self.value)
    def backpropergationnode(self,doutdnode):
        pass


class VarNode(OpNode):
    """
    Represents a variable in the computation graph.
    """
    def __init__(self, varname):
        super().__init__([])
        self.varname = varname

    def __repr__(self):
        return f"Var({self.varname})"
    def nodedef(self,parentnames,small=False):
        if small:return str(self.varname)
        return self.__repr__()
    def _copy_op(self):
        return VarNode(self.varname)
    def signature(self):
        return super().signature(args=self.varname)
    def backpropergationnode(self,doutdnode):
        pass


# Operation-Specific Nodes
class AddNode(OpNode):
    """
    Represents an addition operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "("+"+".join(parentnames)+")"
        return super().nodedef(parentnames)
    def signature(self):
        return super().signature(associative=True)
    def consteliminationnode(self):
        s=0
        parentsnew=[]
        containsconst=False
        for p in self.parents:
            if isinstance(p,ConstNode): 
                s+=p.value
                containsconst=True
            else:
                parentsnew.append(p)
        if len(parentsnew)==0:
            return ConstNode(s)
        if s!=0:
            parentsnew.append( ConstNode(s))
        if len(parentsnew)==1:
            return parentsnew[0]
        if not containsconst:
            return self
        return AddNode(parentsnew)
    def backpropergationnode(self,doutdnode):
        doutdself=doutdnode[self]
        for p in self.parents:
            doutdnode[p]+=doutdself#*one

class MulNode(OpNode):
    """
    Represents a multiplication operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "("+"*".join(parentnames)+")"
        return super().nodedef(parentnames)
    def signature(self):
        return super().signature(associative=True)
    def consteliminationnode(self):
        s=1
        parentsnew=[]
        containsconst=False
        for p in self.parents:
            if isinstance(p,ConstNode): 
                s*=p.value
                containsconst=True
            else:
                parentsnew.append(p)
        if len(parentsnew)==0 or s==0:
            return ConstNode(s)
        if s!=1:
            parentsnew.append( ConstNode(s))
        if len(parentsnew)==1:
            return parentsnew[0]
        if not containsconst:
            return self
        return MulNode(parentsnew)
    def backpropergationnode(self,doutdnode):

        #e=a*b*c*d
        #doutdnode[a]+=1 * b*c*d*1=left[0]*right[-1]
        #doutdnode[b]+=1*a * c*d*1=left[1]*right[-2]
        #doutdnode[c]+=1*a*b * d*1=left[2]*right[-3]
        #doutdnode[d]+=1*a*b*c * 1=left[3]*right[-4]
    
        left=[ConstNode(1)]
        right=[ConstNode(1)]
        for i in range(len(self.parents)-1):
            left.append(left[-1]*self.parents[i])
            right.append(right[-1]*self.parents[-1-i])
        doutdself=doutdnode[self]
        for i in range(len(self.parents)):
            doutdnode[self.parents[i]]+=left[i]*right[-i-1]*doutdself

class SubNode(OpNode):
    """
    Represents a subtraction operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "("+"-".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[1],ConstNode):
            if self.parents[1].value==0:
                return self.parents[0]
            if isinstance(self.parents[0],ConstNode): 
                return ConstNode(self.parents[0].value-self.parents[1].value)
            #return MulNode([self.parents[0],1/self.parents[1].value])
        if isinstance(self.parents[0],ConstNode) and self.parents[0].value==0: 
            return NegNode([self.parents[1]])
        return self
    def normalizenode(self):
        return AddNode([self.parents[0],MulNode([self.parents[1],-1])])
    
class AbsNode(OpNode):
    def nodedef(self,parentnames,small=False):
        if small:return "abs("+"?".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[0],ConstNode):
            return ConstNode(abs(self.parents[0].value))
        return self
    def backpropergationnode(self, doutdnode):
        parent=self.parents[0]
        doutdnode[parent] += doutdnode[self]*SignNode([parent])  
class SignNode(OpNode):
    def nodedef(self,parentnames,small=False):
        if small:return "sign("+"?".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[0],ConstNode):
            v=self.parents[0].value

            return ConstNode(1 if v > 0 else -1 if v < 0 else 0)
        return self
    def backpropergationnode(self, doutdnode):
        # Signum has no meaningful gradient for backpropagation
        # Typically, we stop the gradient here or pass zero
        #doutdnode[self] += ConstNode(0)  # Gradient is zero everywhere except at undefined points
        pass
class SqrtNode(OpNode):
    def nodedef(self,parentnames,small=False):
        if small:return "sqrt("+"?".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[0],ConstNode):
            return ConstNode((self.parents[0].value)**.5)
        return self
    def backpropergationnode(self, doutdnode):
        #sqrt(x)->1/(2*sqrt(x))
        # self.parents[0] ist die Eingabe x der Inversion
        parent = self.parents[0]
        grad = InvNode([2*SqrtNode(parent)])
        doutdnode[parent] += doutdnode[self]*grad
class NegNode(OpNode):
    """
    Represents a subtraction operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "-("+"?".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[0],ConstNode):
            return ConstNode(-self.parents[0].value)
        return self
    def normalizenode(self):
        return MulNode([self.parents[0],-1])
class InvNode(OpNode):
    """
    Represents a division operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "1/("+"?".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[0],ConstNode):
            return ConstNode(1/self.parents[0].value)
        return self
    def backpropergationnode(self, doutdnode):
        # self.parents[0] ist die Eingabe x der Inversion
        parent = self.parents[0]
        grad = ConstNode(-1)  * InvNode([parent * parent])
        doutdnode[parent] += doutdnode[self]*grad
class DivNode(OpNode):
    """
    Represents a division operation in the computation graph.
    """
    def nodedef(self,parentnames,small=False):
        if small:return "("+"/".join(parentnames)+")"
        return super().nodedef(parentnames)
    def consteliminationnode(self):
        if isinstance(self.parents[1],ConstNode):
            if self.parents[1].value==1:
                return self.parents[0]
            if isinstance(self.parents[0],ConstNode): 
                return ConstNode(self.parents[0].value/self.parents[1].value)
            #return MulNode([self.parents[0],1/self.parents[1].value])
        return self
    def normalizenode(self):
        return MulNode([self.parents[0],InvNode([self.parents[1]])])
    def backpropergationnode(self,doutdnode):
        doutdself=doutdnode[self]
        a=self.parents[0]
        b=self.parents[1]
        doutdnode[a]+=doutdself/b
        doutdnode[b]+=-doutdself*a/(b*b)


def simplify(endpoint):
    if isinstance(endpoint,list):
        return simplify(EndpointNode(endpoint)).parents
    if not isinstance(endpoint,OpNode):
        raise ValueError("input must be a list of nodes or a singular node")
    if not isinstance(endpoint,EndpointNode):
        return simplify([endpoint])[0]


    endpoint.replacenode(lambda x:x)#subexpressionelimination
    endpoint.replacenode(lambda x:x.normalizenode())#normlization
    endpoint.replacenode(lambda x:x.consteliminationnode())#remove constant expressions
    endpoint.mergenodes()#merges nodes eg (a+(b+c)) to (a+b+c)
    endpoint.replacenode(lambda x:x.consteliminationnode())# 
    return endpoint

if __name__=="__main__":
    import sys

    # Add your desired path
    sys.path.append('./')
    import cProfile
    import pstats
    from io import StringIO
    print("go")

    # Usage Example
    # x = VarNode("x")

    def fib(n):
        if n==1 or n==2:return ConstNode(1)
        return fib(n-1)+fib(n-2)



    # with cProfile.Profile() as profiler:
    #     c=fib(30)
    #     e=EndpointNode([c]).copy_graph(copydown=False)
    #     #print(e.asplanstr())
    #     print(len(e.topological_sort()))
    #     e.subexpressionelimination()
    #     print(e.asplanstr())

    # # Output the profiling results
    # s = StringIO()
    # ps = pstats.Stats(profiler, stream=s)
    # ps.strip_dirs().sort_stats("tottime").print_stats(20)  # Top 20 by cumulative time
    # print(s.getvalue())

    import algebra.dcga as dcga

    xyz=VarNode("x"),VarNode("y"),VarNode("z")
    point=dcga.point(*xyz)
    torus=dcga.toroid(2,0.5)
    iprod=point.inner(torus)
    
    e=sum(abs(x) for x in iprod.blades.values())
    ep=EndpointNode([e])
    
    ep.replacenode(lambda x:x.normalizenode())
    ep.replacenode(lambda x:x.consteliminationnode())#simplify
    ep.mergenodes()
    print(ep.asplanstrcompact())
    bp=ep.backpropergation()
    e=EndpointNode([bp[v] for v in xyz]+[e])

    #e.subexpressionelimination()
    # import sys
    # sys.setrecursionlimit(10000)
    #e=EndpointNode([fib(30),e])
    #print(len(e.topological_sort()))
    print(e.maxdepth())
    #e.simplify()
    e.replacenode(lambda x:x)#subexpressionelimination
    e.replacenode(lambda x:x.normalizenode())#normlization
    e.replacenode(lambda x:x.consteliminationnode())#simplify
    e.mergenodes()
    e.replacenode(lambda x:x.consteliminationnode())

    def replacer(x):
        d={"x":2,"y":3,"z":4}
        if isinstance(x,VarNode):
            return ConstNode(d.get(x.varname,0))
        return x
    #e.replacenode(replacer)
    #e.replacenode(lambda x:x.constelimination())
    print(e.asplanstr())
    print(e.asplanstrcompact(True))
    # a,b=1,1
    # for i in range(30):
    #     print(b)
    #     a,b=a+b,a
        
    # print((-ConstNode(1)).asplanstr())
    print(calls)

    iprod=dcga.point(2,3,4)^torus
    print(iprod.blades.values())
