import numpy as np
from scipy.spatial import Delaunay

from graph_networks.timing import timing

class Polygon:
    def __init__(self, points):
        # points: point clound who's boundary defines this polygon

        #get the edges that define the polygon's boundary
        self.edges= self.calculate_boundary(points)
        self.edges= self.sort_edges(self.edges)

        self.vertices= self.get_vertices(points)
        self.incident_vertices= points[self.edges].reshape(-1, 4)


    @staticmethod
    def calculate_boundary(points, alpha=10.0, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points.
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
        Source: https://stackoverflow.com/questions/45226931/triangulate-a-set-of-points-with-a-concave-domain
        """
        assert points.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        tri = Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(np.abs(s * (s - a) * (s - b) * (s - c)))

            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
        edges= np.array(list(edges))
        
        return edges

    def sort_edges(self, edges):
        sorted_edges= []
        senders, receivers = edges[:,0], edges[:,1]
        sorted_edges.append(edges[0])
        for _ in range(len(edges)-1):
            s,r= sorted_edges[-1]
            sorted_edges.append(edges[r==senders].flatten())

        return np.array(sorted_edges)
    
    def get_vertices(self, points):

        vertices= points[self.edges[:,0]]
        vertices= np.append(vertices, vertices[[0]], axis=0)
        return vertices

    #todo give credits
    # @timing 
    def inside_polygon(self, point):
        """
        Using horzontal ray algorithm to decide if the given point is inside the polygon
        For readability look at the for loop version of this function (inside_polygon2)
        """
        # get all the vertices higher than the point
        # important the vertices showld form a closed loop 
        yflags= self.vertices[:,1] >= point[1]
        yflags= np.vstack((yflags[:-1],yflags[1:]))

        yflag= yflags[0]!=yflags[1]
        #filter edges whos vertices are above and below the point respectively
        edges_filtered= self.incident_vertices[yflag]

        # for the case above find the point of intersection with x-axis
        xp= edges_filtered[:,2]-point[0]
        yp= edges_filtered[:,3]-point[1]

        x1x2= edges_filtered[:,0]-edges_filtered[:,2]
        y1y2= edges_filtered[:,1]-edges_filtered[:,3]

        inside_flag= yp*x1x2 >= xp*y1y2
        #if the point of intersection is on the right of the point then make this flag true
        inside_flag= inside_flag == yflags[1][yflag]

        #if the number of intersections is odd then the point is inside
        if np.sum(inside_flag)%2 != 0:
            return True
        else:
            return False
    
    def find_mask(self, points):
        mask= np.empty(len(points), dtype=np.bool8)
        for i, point in enumerate(points):
            mask[i]= self.inside_polygon(point)
        return mask





# @timing
def inside_polygon2(pgon, point):
    x,y = point[0], point[1]
    vtx0= pgon[-1]
    yflag0= vtx0[1] >= y 
    vtx1= pgon[0]

    inside_flag= False
    for j in range(len(pgon)):
        yflag1= vtx1[1] >= y

        if yflag0 != yflag1:
            if ((vtx1[1] - y)*( vtx0[0]-vtx1[0]) >= ((vtx1[0]-x)*(vtx0[1]-vtx1[1]))) == yflag1:
                inside_flag= not inside_flag
        if j+1 != len(pgon):
            yflag0= yflag1
            vtx0= vtx1
            vtx1= pgon[j+1]

    return inside_flag    




if __name__=="__main__":

    #testing with known input
    pgon= np.array([[0,0], [1,0], [1,1], [0,1]]) 
    poly= Polygon(pgon)
    point= np.array([0.5, 0.5])
    assert poly.inside_polygon(point) == inside_polygon2(poly.vertices, point)

    # testing with random input
    for _ in range(50):
        pgon= np.random.rand(100,2)*4
        point= np.random.rand(2)*4
        poly= Polygon(pgon)
        
        assert poly.inside_polygon(point) == inside_polygon2(poly.vertices, point)
    

