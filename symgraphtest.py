import aesara
import aesara.tensor as at
from aesara.printing import debugprint

# Define symbolic variables
x = at.scalar('x')
y = at.scalar('y')
z = at.scalar('z')
a = at.scalar('a')

# Define a function f(x, y, z, a)
y1 = x + y + z + a
y2 = x * y - z * a
outputs = at.stack([y1, y2])

# Compile the function
f = aesara.function([x, y, z, a], outputs)

# Inspect the graph
print("Graph Structure:")
debugprint(f)

# Visualize with pydot (optional)
from aesara.d3viz import d3viz
d3viz(f, 'graph.html')  # Generates an interactive HTML visualization
