# Mesh Simplification for general N dimensional vertex data

This is a simple implementation of the [Quadric Error Metric](https://en.wikipedia.org/wiki/Quadric#Quadric_error_metric) for mesh simplification. It is based on the paper [Simplifying Surfaces with Color and Texture using Quadric Error Metrics](https://www.cs.cmu.edu/~garland/Papers/quadric2.pdf) by Michael Garland and Paul S. Heckbert.

## Usage
For the moment it only works with the test, but adding additional data is really simple. To see this tests you can run either of the following commands:
```bash
python3 color_test.py
python3 normal_test.py
python3 scalar_test.py
```

Using the command:
```bash
python3 main.py
```

Will only use the 3D coordinates of the vertex to simplify the mesh. Sample data can be found in the `data` folder and the results will be saved in the `output` folder, which is need to exist beforehand.

## Implementation
Most of the implementation is based on the source code of the [Auxiliar 6](https://colab.research.google.com/drive/1kiHo4fWukfFiGfG2jK7W3IMqZBmZkm1k?usp=sharing#scrollTo=vxPrJ0OmoRbe) of CC5513, so we only will explain the parts that are different.

### Quadrics
First of all, in the original version, the quadric is represented using a 4x4 matrix and vertex are transform to its homogeneous counterpart, but in N dimension we can't do that, so instead we created a new class named Quadric, which store the parameters A, b and c. In addition, this class has functionality for adding two quadrics and for evaluating the error of a vertex. Also, we provide a function that calculate the quadric for a given point and its plane orthogonal vectors (e1 and e2).

### Simplification
With this in mind, the new function to simplify the mesh takes and additional parameter, the data of the points, which has to be the coordinates of the point follow by all the extra data that is need to be considered. It is mandatory that the points has the same index in the mesh and in the data.

The simplification process is the same as the original but no relies on the normal of the faces, so use a new function `compute_face_base` which return the tuple `(p, e1, e2)` needed to calculate the quadric. All the other functions are adapted to consider the points' parameters.

Finally, in order to keep consistent the simplified points, we use a hash map to store the new data with the 3D coordinates as key.
Then with the simplified mesh, we can properly set the data with its corresponding index.

### Face base
This function is a 1-to-1 transcription of the calculation of the plane base `(e1, e2)` from the paper.

## Tests
For the tests we use the following conditions:
1. **Color test**
    
    This test uses the function: `F(x, y, z) = (y, x, x^2 + y^2) = (r, g, b)`, then its normalized to be in the ranged of 0.0 to 1.0
2. Normal test.
    
    This tests uses the normal of each vertex, which is calculated uses the openmesh built-in functionalities.
3. Scalar test.

    This test uses the function: `f(x, y, z) = x^2 + y^2 + z^2`