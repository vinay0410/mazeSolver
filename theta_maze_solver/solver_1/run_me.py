import numpy as np
import cv2
import math
import sys

##  Returns sine of an angle.
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

##  Reads an image from the specified filepath and converts it to Grayscale. Then applies binary thresholding
##  to the image.
def readImage(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,127,255,cv2.THRESH_BINARY)
    return binaryImage

##  This function accepts the img, level and cell number of a particular cell and the size of the maze as input
##  arguments and returns the list of cells which are traversable from the specified cell.
def findNeighbours(img, level, cellnum, size):
    neighbours = []
    ############################# Add your Code Here ################################
    levels = [1, 6, 12, 24 , 24, 24, 48]

    theta = 360.0/levels[level]

    if level == 0:
        edge = img[39:42, 0:theta]
        for i in range(0, 6):
            if cv2.countNonZero(edge[0:3, i*theta/6:(i+1)*theta/6]) > theta/3:
                neighbours.append((1,i+1)) 
            
    else:

        ### For cells other than (0, 0) ##


        ## Extracting the cell from polar image ##
        
        cell = img[level*40:(level+1)*40+2, (cellnum-1)*theta:cellnum*theta+2]
        h, w = cell.shape

        ## Checking for each edge from top edge to left in clockwise direction ##


        ## Top Edge ##
        
        edge = cell[0:3, 0:w] ## Calculating no of white pixels in three different strips of the edge
        if min(cv2.countNonZero(edge[0:1,:]), cv2.countNonZero(edge[1:2,:]), cv2.countNonZero(edge[2:3,:])) > theta/2:
            if level == 1:
                neighbours.append((0, 0))
            elif (level == 5) or (level == 4):
                neighbours.append((level-1, cellnum))
            else:
                neighbours.append((level-1, (cellnum+1)/2))


        # Right Edge ##

        edge = cell[0:h, w-3:w]
        if min(cv2.countNonZero(edge[:,0:1]), cv2.countNonZero(edge[:,1:2]), cv2.countNonZero(edge[:,2:3])) > 20:
            if cellnum == levels[level]:
                neighbours.append((level, 1))
            else:
                neighbours.append((level, cellnum+1))



        ## Checking if its a 5 edge or 4 edge cell ##

        ## 5 edge cells ##
        if (level==1) or (level==2) or (level==5):

            ## Bottom right Edge ##
            edge = cell[h-3:h, w/2:w]
            if min(cv2.countNonZero(edge[0:1,:]), cv2.countNonZero(edge[1:2,:]), cv2.countNonZero(edge[2:3,:])) > theta/4:
                neighbours.append((level+1, cellnum*2))


            ## Bottom Left Edge ##
            edge = cell[h-3:h, 0:w/2]
            if min(cv2.countNonZero(edge[0:1,:]), cv2.countNonZero(edge[1:2,:]), cv2.countNonZero(edge[2:3,:])) > theta/4:
                neighbours.append((level+1, cellnum*2 -1))

        ## 4 edge cells ##
        else:

            ## Bottom Edge ##
            edge = cell[h-3:h, 0:w]
            if min(cv2.countNonZero(edge[0:1,:]), cv2.countNonZero(edge[1:2,:]), cv2.countNonZero(edge[2:3,:])) > theta/2:
                neighbours.append((level+1, cellnum))


        ## Right Edge ##
        edge = cell[0:40, 0:3]
        if min(cv2.countNonZero(edge[:,0:1]), cv2.countNonZero(edge[:,1:2]), cv2.countNonZero(edge[:,2:3])) > 20:
            if (cellnum == 1):
                neighbours.append((level, levels[level]))
            else:
                neighbours.append((level, cellnum-1))



    	

    #################################################################################
    return neighbours

##  colourCell function takes 5 arguments:-
##            img - input image
##            level - level of cell to be coloured
##            cellnum - cell number of cell to be coloured
##            size - size of maze
##            colourVal - the intensity of the colour.
##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################

    if size == 1:
        mid = 220
    else:
        mid = 300

    levels = [1, 6, 12, 24, 24, 24, 48]
    theta = 360.0/levels[level]

    ## Creating a image with white background
    mask = np.zeros((mid*2, mid*2), np.uint8)
    white_img = cv2.bitwise_not(mask)


    if level == 0:      ## Simple circle for centremost cell
        cv2.circle(white_img, (mid, mid), 41 ,colourVal, -1)
        

    elif level == 6:    ## Drawing polygon for cells in level 6


        ## Calculating Point Array ##
        
        points = [(300+(240*cosine((cellnum-1)*theta)), 300+(240*sine((cellnum-1)*theta))), (300+(240*cosine(cellnum*theta)), 300+(240*sine(cellnum*theta))), (300+(280*cosine(cellnum*theta)), 300+(280*sine(cellnum*theta))), (300+(280*cosine((cellnum-1)*theta)), 300+(280*sine((cellnum-1)*theta)))]
        for i in range(0, len(points)):
            points[i] = (int(round(points[i][0])), int(round(points[i][1])))
                        

        cv2.fillConvexPoly(white_img, np.array(points), colourVal)


    else:

        ## Elliptical Arc for the rest of the cell ##
        cv2.ellipse(white_img, (mid, mid), (level*40 + 40, level*40 + 40), 0, (cellnum-1)*theta, cellnum*theta, colourVal, -1)
        cv2.ellipse(white_img, (mid, mid), (level*40-1, level*40-1), 0, (cellnum-1)*theta-2, cellnum*theta+2, 255, -1)


    ## Bitwise and to preserve the black walls from being coloured
    
    cv2.bitwise_and(white_img, img, img)


    #################################################################################
    return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph(img, size):   ## You can pass your own arguments in this space.
    graph = {}
    ############################# Add your Code Here ################################

    ## Mapping image to polar coordinates ##

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, binary = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY)

    if size == 1:
        radius = 200 + 3
        mid = 220
    else:
        radius = 280 + 3
        mid = 300

    ## Creating a white image to map on ##
    polar = np.zeros((radius, 364), np.uint8)
    polar = cv2.bitwise_not(polar)

    ## Mapping values ##
    for theta in range(0, 363):
        sin = sine(theta)
        cos = cosine(theta)
        for r in range(0, radius):
            polar[r, theta] = binary[r*sin + mid, r*cos + mid]

    ## Building Graph ##
       
    if size == 1:
        max_levels = 4
    else:
        max_levels = 6
        
    levels = [1, 6, 12, 24, 24, 24, 48]

    ## Iterating every cell to figure out neighbours and build graph ##    
    for level in range(0, max_levels+1):
        if level == 0:
            point = (0,0)
            neigh = findNeighbours(polar, 0, 0, size)
            graph[point] = neigh
            continue
        for cellnum in range(1, levels[level] + 1):
            point = (level, cellnum)
            neigh = findNeighbours(polar, level, cellnum, size)
            graph[point] = neigh
	
    #################################################################################
    return graph


##  Function accepts some arguments and returns the Start coordinates of the maze.
def findStartPoint(graph, size):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    if size == 1:
        point = (4, 24)
    else:
        point = (6, 48)

    ## Iterating over all cells of the outermost level ##        
    for cell in range(1, point[1]+1):
        list_cells = graph[(point[0], cell)]

        ## If a cell has a neighbour of a level, greater than the outermost level then that's the start point
        if (point[0] +1) in  [x[0] for x in list_cells]: 
            break
    
    start = (point[0], cell)
    
    #################################################################################
    return start

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.
def findPath(graph, start, end, path=[]):      ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    ## Figuring shortest path by traversing graph ##
    
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    shortest = None
    for cell in graph[start]:
        if cell not in path:
            newpath = findPath(graph, cell, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath

    #################################################################################
    return shortest

##  This is the main function where all other functions are called. It accepts filepath
##  of an image as input. You are not allowed to change any code in this function. You are
##  You are only allowed to change the parameters of the buildGraph, findStartPoint and findPath functions
def main(filePath, flag = 0):
    img = readImage(filePath)     ## Read image with specified filepath
    if len(img) == 440:           ## Dimensions of smaller maze image are 440x440
        size = 1
    else:
        size = 2
    maze_graph = buildGraph(img, size)   ## Build graph from maze image. Pass arguments as required
    start = findStartPoint(maze_graph, size)  ## Returns the coordinates of the start of the maze
    shortestPath = findPath(maze_graph, start, (0,0), path=[])  ## Find shortest path. Pass arguments as required.
    print shortestPath
    string = str(shortestPath) + "\n"
    for i in shortestPath:               ## Loop to paint the solution path.
        img = colourCell(img, i[0], i[1], size, 215)
    if __name__ == '__main__':     ## Return value for main() function.
        return img
    else:
        if flag == 0:
            return string
        else:
            return graph
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filepath = sys.argv[1]     ## File path for test image
    img = main(filepath)          ## Main function call
    cv2.imshow("canvas",img)
    cv2.imwrite("solution1.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
