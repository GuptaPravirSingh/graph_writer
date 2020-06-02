# graph_writer


This repository is intended to visualize and publish interactive computation graphs such as - <b>PyTorch Networks</b>
and <b>Tensorflow Networks</b>.

## installation
pip install graph-writer

## usage Example (PyTorch)
```python
    from graph_writer import graph_writer
    import torch
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            with graph_writer.ModuleSpace() as l1:  # Assigns a random new color to all nodes inside this namespace
                self.fc2 = graph_writer.CallWrapper(torch.nn.Linear(120, 84))

            with graph_writer.ModuleSpace() as l2:
                self.fc3 = graph_writer.CallWrapper(torch.nn.Linear(84, 10))
            # If you do not wish to clutter the graph with trivial nodes like relu, then simply do not add a tracer
            # self.relu = CallWrapper(F.relu, node_tracing_name='relu')
            self.add = CallWrapper(torch.add, node_tracing_name='add')

        def forward(self, x):
            x = self.fc2(x)
            
            # You can choose to use a traced node or simply an in built one, like F.relu
            # x1 = self.relu(x)
            x1 = F.relu(x)
            
            x1.node_tracing_name = self.fc2._self_node_tracing_name
            x = self.add(x, x1)
            x = self.fc3(x)
            return x

        def custom_method(self):
            print('Called custom method successfully.')


    net = Net()  # Initialize the network
    
    # Provide the network and the input it takes 'torch.zeros((120,))' in this case for tracing to begin. 
    # This will make one forward pass through the network.
    draw(net, file_name='./my_graph.png', canvas_size_100s_px=(16, 38), torch.zeros((120,)))
    # plt.imsave('./my_graph.png')
    plt.show()
```
You will recieve a `<filename>.html` file in the same directory where you save the png image. This is an interactive version of the network feel fre to move around the nodes so you feel comfortable understanding the architecture.

# warning
PyTorch's`nn.Dataparallel` builds copies of network provided to it by navigating through the properties of the provided network so the tracing code gets stripped away. Hence if you use this library with dataparallel you will not see any connectivity. Simply call the draw functionality before calling dataparallel. Its shown in an example nbelow.
``` python
    from graph_writer import graph_writer
    
    generator = Generator()  # instantiate the model
    
    # Finish drawing
    graph_writer.draw(generator, f'<my_awesome_GAN_generator_ever>.png', (16, 38),
                      torch.zeros((2, 3, 2, 2)))

    # only after drawing proceed with dataparallel
    generator = nn.DataParallel(generator).cuda()
```

## Aknowledgements 

Equal Contributors: [Partha Ghosh](https://github.com/ParthaEth), [Pravir Singh Gupta](https://github.com/GuptaPravirSingh)
