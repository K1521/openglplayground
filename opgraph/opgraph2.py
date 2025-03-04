calls=0
class OpNode:
    """
    Base class for all operation nodes in the computation graph.
    """
    def __init__(self, parents):
        self.parents = [OpNode.nodify(p) for p in parents]
       

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
    
    
    def topological_sort(self, visited=None, ordering=None):
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
    def __radd__(self, other):
        return AddNode([other, self])

    def __mul__(self, other):
        """Overload the * operator."""
        return MulNode([self, other])
    def __rmul__(self, other):
        """Overload the * operator."""
        return MulNode([other,self])

    def __sub__(self, other):
        """Overload the - operator."""
        return SubNode([self, other])
    def __rsub__(self, other):
        """Overload the - operator."""
        return SubNode([other,self])

    def __truediv__(self, other):
        """Overload the / operator."""
        return DivNode([self, other])
    def __neg__(self):
        return NegNode([self])

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
    def constelimination(self):
        return self
    
    def simplify(self,nodesigcash=None,nodecash=None,issimplified=False):
        global calls
        calls+=1
        if calls==20000:
            pass
        #if children is None:children=dict()#node->set(node.children)
        if nodesigcash is None:nodesigcash=dict()#node.signature()->node
        if nodecash is None:nodecash=dict()
        #if children is None:children=dict()
        #if visited is None:visited=set()#node #can replaced by children dict

        cashednode=nodecash.get(self,None)
        if cashednode is not None:return cashednode

        self.parents=[p.simplify(nodesigcash,nodecash)  for p in self.parents]
        sig=self.signature()
        cashednode=nodesigcash.get(sig,None)
        if cashednode is not None:return cashednode

        if issimplified:
            simplified=self
        else:
            simplified=self.constelimination()
            if simplified!=self:
                simplified=simplified.simplify(nodesigcash,nodecash,issimplified=True)


        nodecash[self]=simplified
        nodesigcash[sig]=simplified

        return simplified



class EndpointNode(OpNode):
    """
    Represents an addition operation in the computation graph.
    """
    def subexpressionelimination(self):
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
    
 
    
    
    # def simplify(self):
    #     nodes,children=self.topsortwithchildren()

    #     sigtonode=dict()#repr->node
    #     for n in nodes:
    #         sig=n.signature()
    #         cashednode=sigtonode.get(sig,None)
    #         if cashednode is None:#if the node isnt cashed
    #             nsimplified=n.constelimination()
    #             #nsimplified.parents=
                
    #             sigsimplified=nsimplified.signature()

    #             if sigsimplified==ConstNode(16).signature():
    #                 pass

    #             cashednode=sigtonode.get(sigsimplified,None)
    #             if cashednode is None:#if simplified and node isnt cashed
    #                 sigtonode[sigsimplified] = nsimplified
    #                 sigtonode[sig] = nsimplified
    #             else:
    #                 #sigtonode[sigsimplified] = cashednode #already in cash
    #                 sigtonode[sig] = cashednode


    #         if cashednode is not None:#replace n with cashed
    #             for c in children[n]:
    #                 c.parents= [cashednode if p is n else p for p in c.parents]
    #             #children[samenode]+=children[n]#not neccesary
    #             #del children[n]
    #     return list(sigtonode.values())    
   

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
    def constelimination(self):
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

class MulNode(OpNode):
    """
    Represents a multiplication operation in the computation graph.
    """

    # def __repr__(self):
    #     return f"Mul({self.parents[0]}, {self.parents[1]})"
    def signature(self):
        return super().signature(associative=True)
    def constelimination(self):
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

class SubNode(OpNode):
    """
    Represents a subtraction operation in the computation graph.
    """

    def __repr__(self):
        return f"Sub({self.parents[0]}, {self.parents[1]})"
    def constelimination(self):
        if isinstance(self.parents[1],ConstNode):
            if self.parents[1].value==0:
                return self.parents[0]
            if isinstance(self.parents[0],ConstNode): 
                return ConstNode(self.parents[0].value-self.parents[1].value)
            #return MulNode([self.parents[0],1/self.parents[1].value])
        if isinstance(self.parents[0],ConstNode) and self.parents[0].value==0: 
            return NegNode([self.parents[1]])
        return self
    
class NegNode(OpNode):
    """
    Represents a subtraction operation in the computation graph.
    """

    def __repr__(self):
        return f"Neg({self.parents[0]})"
    def constelimination(self):
        if isinstance(self.parents[0],ConstNode):
            return ConstNode(-self.parents[0].value)
        return self


class DivNode(OpNode):
    """
    Represents a division operation in the computation graph.
    """
    def __repr__(self):
        return f"Div({self.parents[0]}, {self.parents[1]})"
    def constelimination(self):
        if isinstance(self.parents[1],ConstNode):
            if self.parents[1].value==1:
                return self.parents[0]
            if isinstance(self.parents[0],ConstNode): 
                return ConstNode(self.parents[0].value/self.parents[1].value)
            #return MulNode([self.parents[0],1/self.parents[1].value])
        return self

import cProfile
import pstats
from io import StringIO

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


point=dcga.point(VarNode("x"),VarNode("y"),VarNode("z"))
torus=dcga.toroid(2,0.5)
iprod=point^torus
e=EndpointNode(iprod.blades.values())
#e.subexpressionelimination()
# import sys
# sys.setrecursionlimit(10000)
e=EndpointNode([fib(30),e])
print(len(e.topological_sort()))
print(e.maxdepth())
e.simplify()
#e.subexpressionelimination()
print(e.asplanstr())

# a,b=1,1
# for i in range(30):
#     print(b)
#     a,b=a+b,a
    
# print((-ConstNode(1)).asplanstr())
print(calls)


